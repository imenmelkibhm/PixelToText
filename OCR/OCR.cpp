// (c) 2013 ABBYY Production LLC 
// SAMPLES code is property of ABBYY, exclusive rights are reserved.
//
// DEVELOPER is allowed to incorporate SAMPLES into his own APPLICATION and modify it under
// the  terms of  License Agreement between  ABBYY and DEVELOPER.

// ABBYY FineReader Engine 11.1 Sample

// This sample shows basic steps of ABBYY FineReader Engine usage:
// Initializing, opening image file, recognition and export.

#include "FREngineLoader.h"
#include "BstrWrap.h"
#include "AbbyyException.h"
#include "../SamplesConfig.h"
#include "OCR.h"
#include <string>
#include <iostream>

// Forward declaration
void displayMessage( const wchar_t* text );
BSTR processImage(BSTR imagePath);



extern "C" BSTR executeTask()
{
    BSTR retxt = new wchar_t[256];
	try {
		// Load ABBYY FineReader Engine
		displayMessage( L"Initializing Engine..." );
		LoadFREngine();

		// Process the image
		CBstr imagePath = "/opt/exe/PixelToText/Debug_BFMTV_20161018_02275507_02325507_SD/zone3425_0_zoomed.jpg";
		retxt = processImage(imagePath);
		displayMessage( L"Loading image22..." );
		//std::wstring widestr = std::wstring(retxt.begin(), retxt.end());
		displayMessage( retxt);

		// Convert the recognized text
		std::string retext = "";

		// Unload ABBYY FineReader Engine
		displayMessage( L"Deinitializing Engine..." );
		UnloadFREngine();

	} catch( CAbbyyException& e ) {
		displayMessage( e.Description() );
	}
	return retxt;
}

BSTR processImage(BSTR imagePath)
{	
	// Create document from image file
	displayMessage( L"Loading image..." );
	//CBstr imagePath = "/opt/exe/PixelToText/Debug_BFMTV_20161018_02275507_02325507_SD/zone3425_0_zoomed.jpg"; //Concatenate( GetSamplesFolder(), L"/SampleImages/Demo.tif" );
	CSafePtr<IFRDocument> frDocument = 0;
	CheckResult( FREngine->CreateFRDocumentFromImage( imagePath, 0, frDocument.GetBuffer() ) );

	// Recognize document
	displayMessage( L"Recognizing..." );
	CheckResult( frDocument->Process() );

    //CFileWriter *fileWriter =
    //CheckResult( frDocument->ExportToMemory(fileWriter, FEF_RTF, 0));

    std::string txt = "";
    BSTR text;
    CSafePtr<IFRPages> frPages = 0;
    int count = 0;
    frDocument->get_Pages(frPages.GetBuffer());
    frPages->get_Count(&count);
    for(int i = 0; i< count; i++)
    {
        displayMessage( L"getting pages..." );
        CSafePtr<IFRPage> frPage = 0;
        frPages->get_Element(i, &frPage);
        CSafePtr<IPlainText> plainText = 0;
        frPage->get_PlainText(&plainText);

        plainText->get_Text(&text);
        displayMessage(text);
        //CSafePtr<ILayout> layout = 0;
        //CSafePtr<ILayoutBlocks> layoutBlocks = 0;
        //layout->get_Blocks(layoutBlocks.GetBuffer());

    }


	// Save results
	displayMessage( L"Saving results..." );
	CBstr exportPath = Concatenate( GetSamplesFolder(), L"/SampleImages/Demo.rtf" );
	CheckResult( frDocument->Export(  exportPath, FEF_RTF, 0  ) );
	return text;
}

void displayMessage( const wchar_t* text )
{
	wprintf( text );
	wprintf( L"\n" );
}

int main()
{
	//executeTask();
	return 0;
}

