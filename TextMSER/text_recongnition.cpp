/*
 * textdetection.cpp
 *
 * A demo program of End-to-end Scene Text Detection and Recognition:
 * Shows the use of the Tesseract OCR API with the Extremal Region Filter algorithm described in:
 * Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012
 *
 * Created on: Jul 31, 2014
 *     Author: Lluis Gomez i Bigorda <lgomez AT cvc.uab.es>
 */

#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "../OCR/OCR.h"

#include <boost/filesystem.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dlfcn.h>

using namespace std;
using namespace cv;
using namespace cv::text;

//Calculate edit distance nbetween two words
size_t edit_distance(const string& A, const string& B);
size_t min(size_t x, size_t y, size_t z);
bool   isRepetitive(const string& s);
bool   sort_by_lenght(const string &a, const string &b);
//Draw ER's in an image via floodFill
void   er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation);
bool  isBlackTextOnWhite(Mat textZone);

string make_filename( const char* base_dir, const string& sub_dir, int index, const string& ext )
{
  ostringstream result;
  ostringstream result_folder;

  std::string path(base_dir);
  result_folder << path << "/" << sub_dir;
  boost::filesystem::path dir(result_folder.str());
  if (!(boost::filesystem::exists(dir)))
        {
            cout<< "Output OCR Foldernot found..."<<endl;
            cout<< "Creating the output folder.."<< result_folder.str() <<endl;
            boost::filesystem::create_directory(dir);
        }
  result << path << "/" << sub_dir << "/" <<index << ext;
  return result.str();
}

/*Text Regions Detection*/
extern "C"  int text_recognition(unsigned char* img, int rows, int cols, int stime, int Debug, const char *debugPath, int MULTI_CHANNEL)
{

    if (Debug)
    {
        std::string str(debugPath);
        boost::filesystem::path dir(str);
        if (!(boost::filesystem::exists(dir)))
        {
            cout<< "Debug folder not found..."<<endl;
            cout<< "Creating the debug folder.."<< str <<endl;
            boost::filesystem::create_directory(dir);
        }
    }

    // Extract channels to be processed individually
    Mat image(rows, cols, CV_8UC3, (void *) img);

    if (Debug)
    {
        char file_name[100];
        sprintf(file_name, "%s/image_orig%d.jpg", debugPath, stime);
        imwrite(file_name, image);
    }

    vector<Mat> channels;
    if (MULTI_CHANNEL)
      computeNMChannels(image, channels);
    else
    {
        Mat grey; //Use only the gray channel and its opposite
        cvtColor(image,grey,COLOR_RGB2GRAY);
        channels.push_back(grey);
    }

    int cn = (int)channels.size();
    // Append negative channels to detect ER- (bright regions over dark background)
    for (int c = 0; c < cn; c++)
        channels.push_back(255-channels[c]);

    double t_d = (double)getTickCount();
    // Create ERFilter objects with the 1st and 2nd stage default classifiers: classifiers should be located at the start up folder
    Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"),16,0.000015f,0.13f,0.2f,true,0.1f);
    Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"),0.5);

    // Apply the default cascade classifier to each independent channel (could be done in parallel)
    vector<vector<ERStat> > regions(channels.size());
    for (int c=0; c<(int)channels.size(); c++)
    {
        er_filter1->run(channels[c], regions[c]);
        er_filter2->run(channels[c], regions[c]);
    }
    cout << "TIME_REGION_DETECTION = " << ((double)getTickCount() - t_d)*1000/getTickFrequency() << "  FRAME "<< stime << endl;

    Mat out_img_decomposition= Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
    vector<Vec2i> tmp_group;
    for (int i=0; i<(int)regions.size(); i++)
    {
        for (int j=0; j<(int)regions[i].size();j++)
        {
            tmp_group.push_back(Vec2i(i,j));
        }
        Mat tmp= Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
        er_draw(channels, regions, tmp_group, tmp);
        if (i > 0)
            tmp = tmp / 2;
        out_img_decomposition = out_img_decomposition | tmp;
        tmp_group.clear();
    }

    if (Debug)
    {
        char file_name_r[100];
        sprintf(file_name_r, "%s/out_img_decomposition%d.jpg", debugPath, stime);
        imwrite(file_name_r,out_img_decomposition);
    }

    // Detect character groups
    double t_g = (double)getTickCount();
    vector< vector<Vec2i> > nm_region_groups;
    vector<Rect> nm_boxes;
    erGrouping(image, channels, regions, nm_region_groups, nm_boxes,ERGROUPING_ORIENTATION_HORIZ);
    cout << "TIME_GROUPING = " << ((double)getTickCount() - t_g)*1000/getTickFrequency() << endl;

    // Remove too little isolated regions
    for (int i=0; i<(int)nm_boxes.size(); i++)
    {
        float ratio = (float) nm_boxes[i].height/nm_boxes[i].width;
        int height = nm_boxes[i].height;
        if (((ratio < 0.11) && (height < 14)) || (height <= 12))
        {
            nm_boxes.erase(nm_boxes.begin()+i);
            i--;
        }
    }

    // Merge overlapping text regions
    bool foundIntersection = false;
    do
    {
        foundIntersection = false;
        for (int i=0; i<(int)nm_boxes.size(); i++)
        {
            Rect current =  nm_boxes[i];
            for (int j=i+1; j<(int)nm_boxes.size(); j++)
            {
                Rect inter = current & nm_boxes[j]; //compute the rectangles intersection
                if (inter.area() > 0)
                {
                    foundIntersection = true;
                    Rect uni = nm_boxes[i] | nm_boxes[j]; //compute the rectangles union
                    current = uni & Rect(0,0,image.cols, image.rows); // To avoid a rectangle getting out of the image
                    nm_boxes.erase(nm_boxes.begin()+j);
                    nm_boxes.at(i) = current;
                    j--;
                }
            }
        }

    } while (foundIntersection);
    cout << "TIME_MERGING = " << ((double)getTickCount() - t_g)*1000/getTickFrequency() << endl;

    if (Debug)
    {
        Mat rectangles;
        char file_name_[100];
        image.copyTo(rectangles);
        for (int i=0; i<(int)nm_boxes.size(); i++)
        {
            rectangle(rectangles, nm_boxes[i].tl(), nm_boxes[i].br(), Scalar(255,0,255), 2);
            float ratio = (float) nm_boxes[i].height/nm_boxes[i].width;
            //char str[200];
            //sprintf(str,"%d %d ", nm_boxes[i].height, nm_boxes[i].width);
            //putText(rectangles, str, nm_boxes[i].tl(), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(100,50,50), 2, CV_AA );
            sprintf(file_name_, "%s/rectangles_%d.jpg",debugPath, stime);
            imwrite(file_name_,rectangles );
        }
    }

    //Enlarge texte zones to include border charatcers:
    for (int i=0; i<(int)nm_boxes.size(); i++)
    {
        cv::Point  inflationPoint(-5,-5);
        cv::Size  inflationSize(10,10);
        nm_boxes[i] += inflationPoint;
        nm_boxes[i] += inflationSize;
        nm_boxes[i] = nm_boxes[i] & Rect(0,0,image.cols, image.rows); // To avoid a rectangle getting out of the image
    }

    cout << "TIME_FILTERING = " << ((double)getTickCount() - t_g)*1000/getTickFrequency() << endl;
    if (Debug)
    {
        Mat rectangles;
        char file_name_[100];
        image.copyTo(rectangles);
        for (int i=0; i<(int)nm_boxes.size(); i++)
        {
            rectangle(rectangles, nm_boxes[i].tl(), nm_boxes[i].br(), Scalar(255,0,255), 2);
            sprintf(file_name_, "%s/rectangles_%d_filtered.jpg",debugPath, stime);
            imwrite(file_name_,rectangles );
        }
    }

    std::vector<const wchar_t*> textZonesPaths;
    // Post-processing of the text regions to extract the binary text images.
    for (int i=0; i<(int)nm_boxes.size(); i++)
    {
        Mat roi = Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
        image(nm_boxes[i]).copyTo(roi);
        char file_name_[100];
        sprintf(file_name_, "%s/zone%d_%d.jpg",debugPath, stime ,i);
        imwrite(file_name_,roi);
        //Zoom
        Mat resized;
        Mat roi_grey;
        resize(roi, resized, Size(), 2, 2, INTER_CUBIC);
        cvtColor(resized,roi_grey,COLOR_RGB2GRAY);
         // Detect Light text on Dark background or Dark text on light background
         bool text_conf = isBlackTextOnWhite(roi_grey);

        //Binarise the image
        Mat thresholded;
        GaussianBlur(roi_grey,roi_grey, Size(3,3), 0,0);
        cv::threshold(roi_grey,thresholded, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        if (!text_conf)//Inverse the image if the text is in white
            thresholded = 255 - thresholded;

        //erode to get more precise caracters
        erode(thresholded, thresholded, Mat(), Point(-1, -1), 2, 1, 1);

        wchar_t *path = new wchar_t[256];
        swprintf(path,100, L"%s/zone%d_%d_thresh.jpg", debugPath, stime ,i);
        sprintf(file_name_, "%s/zone%d_%d_thresh.jpg", debugPath, stime ,i);
        imwrite(file_name_,thresholded);
        textZonesPaths.push_back(path);
    }


     /*Text Recognition (OCR)*/
    double t_r = (double)getTickCount();
    vector<string> detected_txt;
    cout << "TIME_OCR_INITIALIZATION = " << ((double)getTickCount() - t_r)*1000/getTickFrequency() << endl;
    for (int i=0; i<(int)nm_boxes.size(); i++)
    {
        wchar_t* res = executeTask(textZonesPaths[i]);
        wstring ws(res);
        string output(ws.begin(), ws.end());
        detected_txt.push_back(output);
    }

    //save the text into txt file to be used to create the final xml file
    if (!detected_txt.empty())
    {
        std::ofstream output_file(make_filename( debugPath, "OCR_results",  stime, ".txt" ).c_str());
        std::ostream_iterator<std::string> output_iterator(output_file, "\n");
        std::copy(detected_txt.begin(), detected_txt.end(), output_iterator);
    }

    return 1;



}

bool isBlackTextOnWhite(Mat textZone){

    bool conf = false;
    float Ratio = 1.0;
    int dark=0;
    int light=0;
    int total_number=textZone.rows * textZone.cols;
    // Initialize parameters
    int histSize = 256; //bin size
    float range[] = { 0, 255 };
    const float *ranges[] = { range };

    // Calculate histogram
    MatND hist;
    calcHist( &textZone, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false );
    //threshold the image to 125
    Mat binary_textZone;
    threshold(textZone, binary_textZone, 127, 255, THRESH_BINARY);
    light = countNonZero(binary_textZone);
    dark = total_number - light;

    Ratio = (float)dark/light;
    if (Ratio < 1)
        conf = true;

    if (0)
    {
        cout<< "Light " << light << "  And Dark "<< dark << " Ratio = "<< (float)dark/light<< endl;
        // Plot the histogram
        int hist_w = 512; int hist_h = 400;
        int bin_w = cvRound( (double) hist_w/histSize );

        Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );
        normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

        for( int i = 1; i < histSize; i++ )
        {
          line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                           Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                           Scalar( 255, 0, 0), 2, 8, 0  );
        }

        namedWindow( "Gray", 1 );       imshow( "Gray", textZone );
        namedWindow( "Result", 1 );    imshow( "Result", histImage );
        namedWindow( "Binary", 1 );    imshow( "Binary", binary_textZone);
        waitKey(0);

    }

    return conf;

}

extern "C"  int   text_recognition_tesseract(Mat image, int RecEval, vector<string> words_gt, int num_gt_characters  )
{

 /*Text Detection*/

    // Extract channels to be processed individually
    vector<Mat> channels;

    Mat grey;
    cvtColor(image,grey,COLOR_RGB2GRAY);

    // Notice here we are only using grey channel, see textdetection.cpp for example with more channels
    channels.push_back(grey);
    channels.push_back(255-grey);

    double t_d = (double)getTickCount();
    // Create ERFilter objects with the 1st and 2nd stage default classifiers
    Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("./TextMSER/trained_classifierNM1.xml"),8,0.00015f,0.13f,0.2f,true,0.1f);
    Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("./TextMSER/trained_classifierNM2.xml"),0.5);

    vector<vector<ERStat> > regions(channels.size());
    // Apply the default cascade classifier to each independent channel (could be done in parallel)
    for (int c=0; c<(int)channels.size(); c++)
    {
        er_filter1->run(channels[c], regions[c]);
        er_filter2->run(channels[c], regions[c]);
    }
    cout << "TIME_REGION_DETECTION = " << ((double)getTickCount() - t_d)*1000/getTickFrequency() << endl;

    Mat out_img_decomposition= Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
    vector<Vec2i> tmp_group;
    for (int i=0; i<(int)regions.size(); i++)
    {
        for (int j=0; j<(int)regions[i].size();j++)
        {
            tmp_group.push_back(Vec2i(i,j));
        }
        Mat tmp= Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
        er_draw(channels, regions, tmp_group, tmp);
        if (i > 0)
            tmp = tmp / 2;
        out_img_decomposition = out_img_decomposition | tmp;
        tmp_group.clear();
    }

    double t_g = (double)getTickCount();
    // Detect character groups
    vector< vector<Vec2i> > nm_region_groups;
    vector<Rect> nm_boxes;
    erGrouping(image, channels, regions, nm_region_groups, nm_boxes,ERGROUPING_ORIENTATION_HORIZ);
    cout << "TIME_GROUPING = " << ((double)getTickCount() - t_g)*1000/getTickFrequency() << endl;

    /*Text Recognition (OCR)*/
    double t_r = (double)getTickCount();
    Ptr<OCRTesseract> ocr = OCRTesseract::create();
    cout << "TIME_OCR_INITIALIZATION = " << ((double)getTickCount() - t_r)*1000/getTickFrequency() << endl;
    string output;

    Mat out_img;
    Mat out_img_detection;
    Mat out_img_segmentation = Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
    image.copyTo(out_img);
    image.copyTo(out_img_detection);
    float scale_img  = 600.f/image.rows;
    float scale_font = (float)(2-scale_img)/1.4f;
    vector<string> words_detection;

    t_r = (double)getTickCount();

    for (int i=0; i<(int)nm_boxes.size(); i++)
    {

        rectangle(out_img_detection, nm_boxes[i].tl(), nm_boxes[i].br(), Scalar(0,255,255), 3);

        Mat group_img = Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
        er_draw(channels, regions, nm_region_groups[i], group_img);
        Mat group_segmentation;
        group_img.copyTo(group_segmentation);
        char file_name[100];

        //image(nm_boxes[i]).copyTo(group_img);
        group_img(nm_boxes[i]).copyTo(group_img);
        copyMakeBorder(group_img,group_img,15,15,15,15,BORDER_CONSTANT,Scalar(0));
        sprintf(file_name, "group_img%d.jpg", i);
        imwrite(file_name,group_segmentation );
        vector<Rect>   boxes;
        vector<string> words;
        vector<float>  confidences;
        ocr->run(group_img, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);

        output.erase(remove(output.begin(), output.end(), '\n'), output.end());
        //cout << "OCR output = \"" << output << "\" lenght = " << output.size() << endl;
        if (output.size() < 3)
            continue;

        for (int j=0; j<(int)boxes.size(); j++)
        {
            boxes[j].x += nm_boxes[i].x-15;
            boxes[j].y += nm_boxes[i].y-15;

            //cout << "  word = " << words[j] << "\t confidence = " << confidences[j] << endl;
            if ((words[j].size() < 2) || (confidences[j] < 51) ||
                    ((words[j].size()==2) && (words[j][0] == words[j][1])) ||
                    ((words[j].size()< 4) && (confidences[j] < 60)) ||
                    isRepetitive(words[j]))
                continue;
            words_detection.push_back(words[j]);
            rectangle(out_img, boxes[j].tl(), boxes[j].br(), Scalar(255,0,255),3);
            Size word_size = getTextSize(words[j], FONT_HERSHEY_SIMPLEX, (double)scale_font, (int)(3*scale_font), NULL);
            rectangle(out_img, boxes[j].tl()-Point(3,word_size.height+3), boxes[j].tl()+Point(word_size.width,0), Scalar(255,0,255),-1);
            putText(out_img, words[j], boxes[j].tl()-Point(1,1), FONT_HERSHEY_SIMPLEX, scale_font, Scalar(255,255,255),(int)(3*scale_font));
            out_img_segmentation = out_img_segmentation | group_segmentation;
        }

    }

    cout << "TIME_OCR = " << ((double)getTickCount() - t_r)*1000/getTickFrequency() << endl;

    /* Recognition evaluation with (approximate) hungarian matching and edit distances */

    if(RecEval==1)
    {


        if (words_detection.empty())
        {
            //cout << endl << "number of characters in gt = " << num_gt_characters << endl;
            cout << "TOTAL_EDIT_DISTANCE = " << num_gt_characters << endl;
            cout << "EDIT_DISTANCE_RATIO = 1" << endl;
        }
        else
        {

            sort(words_gt.begin(),words_gt.end(),sort_by_lenght);

            int max_dist=0;
            vector< vector<int> > assignment_mat;
            for (int i=0; i<(int)words_gt.size(); i++)
            {
                vector<int> assignment_row(words_detection.size(),0);
                assignment_mat.push_back(assignment_row);
                for (int j=0; j<(int)words_detection.size(); j++)
                {
                    assignment_mat[i][j] = (int)(edit_distance(words_gt[i],words_detection[j]));
                    max_dist = max(max_dist,assignment_mat[i][j]);
                }
            }

            vector<int> words_detection_matched;

            int total_edit_distance = 0;
            int tp=0, fp=0, fn=0;
            for (int search_dist=0; search_dist<=max_dist; search_dist++)
            {
                for (int i=0; i<(int)assignment_mat.size(); i++)
                {
                    int min_dist_idx =  (int)distance(assignment_mat[i].begin(),
                                        min_element(assignment_mat[i].begin(),assignment_mat[i].end()));
                    if (assignment_mat[i][min_dist_idx] == search_dist)
                    {
                        //cout << " GT word \"" << words_gt[i] << "\" best match \"" << words_detection[min_dist_idx] << "\" with dist " << assignment_mat[i][min_dist_idx] << endl;
                        if(search_dist == 0)
                            tp++;
                        else { fp++; fn++; }

                        total_edit_distance += assignment_mat[i][min_dist_idx];
                        words_detection_matched.push_back(min_dist_idx);
                        words_gt.erase(words_gt.begin()+i);
                        assignment_mat.erase(assignment_mat.begin()+i);
                        for (int j=0; j<(int)assignment_mat.size(); j++)
                        {
                            assignment_mat[j][min_dist_idx]=INT_MAX;
                        }
                        i--;
                    }
                }
            }

            for (int j=0; j<(int)words_gt.size(); j++)
            {
                //cout << " GT word \"" << words_gt[j] << "\" no match found" << endl;
                fn++;
                total_edit_distance += (int)words_gt[j].size();
            }
            for (int j=0; j<(int)words_detection.size(); j++)
            {
                if (find(words_detection_matched.begin(),words_detection_matched.end(),j) == words_detection_matched.end())
                {
                    //cout << " Detection word \"" << words_detection[j] << "\" no match found" << endl;
                    fp++;
                    total_edit_distance += (int)words_detection[j].size();
                }
            }


            //cout << endl << "number of characters in gt = " << num_gt_characters << endl;
            cout << "TOTAL_EDIT_DISTANCE = " << total_edit_distance << endl;
            cout << "EDIT_DISTANCE_RATIO = " << (float)total_edit_distance / num_gt_characters << endl;
            cout << "TP = " << tp << endl;
            cout << "FP = " << fp << endl;
            cout << "FN = " << fn << endl;
        }
    }



    //resize(out_img_detection,out_img_detection,Size(image.cols*scale_img,image.rows*scale_img));
    //imshow("detection", out_img_detection);
    //imwrite("detection.jpg", out_img_detection);
    //resize(out_img,out_img,Size(image.cols*scale_img,image.rows*scale_img));
    namedWindow("recognition",WINDOW_NORMAL);
    imshow("recognition", out_img);
    waitKey(0);
    imwrite("recognition.jpg", out_img);
    imwrite("segmentation.jpg", out_img_segmentation);
    imwrite("decomposition.jpg", out_img_decomposition);


    return 1;
}



size_t min(size_t x, size_t y, size_t z)
{
    return x < y ? min(x,z) : min(y,z);
}

size_t edit_distance(const string& A, const string& B)
{
    size_t NA = A.size();
    size_t NB = B.size();

    vector< vector<size_t> > M(NA + 1, vector<size_t>(NB + 1));

    for (size_t a = 0; a <= NA; ++a)
        M[a][0] = a;

    for (size_t b = 0; b <= NB; ++b)
        M[0][b] = b;

    for (size_t a = 1; a <= NA; ++a)
        for (size_t b = 1; b <= NB; ++b)
        {
            size_t x = M[a-1][b] + 1;
            size_t y = M[a][b-1] + 1;
            size_t z = M[a-1][b-1] + (A[a-1] == B[b-1] ? 0 : 1);
            M[a][b] = min(x,y,z);
        }

    return M[A.size()][B.size()];
}

bool isRepetitive(const string& s)
{
    int count = 0;
    for (int i=0; i<(int)s.size(); i++)
    {
        if ((s[i] == 'i') ||
                (s[i] == 'l') ||
                (s[i] == 'I'))
            count++;
    }
    if (count > ((int)s.size()+1)/2)
    {
        return true;
    }
    return false;
}


void er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation)
{
    for (int r=0; r<(int)group.size(); r++)
    {
        ERStat er = regions[group[r][0]][group[r][1]];
        if (er.parent != NULL) // deprecate the root region
        {
            int newMaskVal = 255;
            int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
            floodFill(channels[group[r][0]],segmentation,Point(er.pixel%channels[group[r][0]].cols,er.pixel/channels[group[r][0]].cols),
                      Scalar(255),0,Scalar(er.level),Scalar(0),flags);
        }
    }
}

bool   sort_by_lenght(const string &a, const string &b){return (a.size()>b.size());}


//Perform text detection and recognition and evaluate results using edit distance
int main(int argc, char* argv[])
{

    cout << endl << argv[0] << endl << endl;
    cout << "A demo program of End-to-end Scene Text Detection and Recognition: " << endl;
    cout << "Shows the use of the Tesseract OCR API with the Extremal Region Filter algorithm described in:" << endl;
    cout << "Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012" << endl << endl;

    Mat image;
    if(argc>1)
    {
        image  = imread(argv[1]);
        cout << "IMG_W=" << image.cols << endl;
        cout << "IMG_H=" << image.rows << endl;
     }
    else
    {
        cout << "    Usage: " << argv[0] << " <input_image> [<gt_word1> ... <gt_wordN>]" << endl;
        return(0);
    }

    int num_gt_characters   = 0;
    vector<string> words_gt;
    if(argc>2)
    {

        for (int i=2; i<argc; i++)
        {
            string s = string(argv[i]);
            if (s.size() > 0)
            {
                words_gt.push_back(string(argv[i]));
                //cout << " GT word " << words_gt[words_gt.size()-1] << endl;
                num_gt_characters += (int)(words_gt[words_gt.size()-1].size());
            }
        }
    }
    text_recognition_tesseract(image, 0, words_gt, num_gt_characters );
    //text_recognition2(image);
    return 0;
}