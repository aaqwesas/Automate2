# PDF Document Processor
Similar to Automate1, this version directly try to extracts the name form the page using OCR and regular expressions. This will splits the original PDFs into individual single-page files named according to the extracted information. Finally, the resulting files are packaged into ZIP archives for easy distribution.

> Note that this project is built for very specific use case only, this is unlikely that it will work for your certificate/recepit that you might want to process.You could try to modify the regular expression to your liking.

## Features

- Processes certificate and receipt PDFs separately
- Uses OCR (via Tesseract) to extract text from PDF pages
- Applies pattern matching to identify participant names:
  - Certificates: Matches text following "to certify that [Name] has..."
  - Receipts: Matches text between "Payer" and "Amount"
- Sanitizes extracted names for safe use as filenames
- Splits multi-page PDFs into individual pages with the extracted name
- Packages output into ZIP files and cleans up intermediate directories
- Parallel processing
- logging to both console and file (`app.log`)

## Requirements

### System Dependencies

- **Poppler**: Required for PDF-to-image conversion
  - Windows: Download from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases/) and add `bin/` to PATH
  - macOS: `brew install poppler`
  - Linux (Debian/Ubuntu): `sudo apt-get install poppler-utils` (based on your distro)

- **Tesseract OCR**:
  - Windows: Download installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
  - macOS: `brew install tesseract`
  - Linux (Debian/Ubuntu): `sudo apt-get install tesseract-ocr` (based on your distro)

### Python Dependencies

Install required packages with:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pdf2image PyPDF2 pytesseract tqdm
```

## Usage

1. Place certificate PDFs in a folder named `Certificates`
2. Place receipt PDFs in a folder named `Receipts`
3. Run the script:

```bash
python main.py
```

4. Enter the course code when prompted

The script will:
- Process all PDFs in both folders
- Create output directories `split_certificates` and `split_receipts`
- Generate ZIP files `split_certificates.zip` and `split_receipts.zip`
- Remove the intermediate output directories after zipping

## Project Structure

```
Automate2/
├── Certificates/          # Input: certificate PDFs
├── Receipts/              # Input: receipt PDFs
├── split_certificates.zip # Output: processed certificates
├── split_receipts.zip     # Output: processed receipts
├── app.log                # Log file
├── requirements.txt       # python requirements
└── main.py                # Main script
```

## Configuration

### Whitelist Characters

The OCR process restricts output to a predefined set of characters (letters, common accented characters, spaces, and basic punctuation). Modify the `WHITELIST` constant in the script to adjust allowed characters.

### Concurrency

The script automatically configures thread count based on CPU cores (minimum 32 workers). Adjust `MAX_WORKERS` if needed for your system.

### Logging

Logs are written to both console and `app.log` with timestamps. Log level is set to INFO by default.

## Limitations

- Name extraction accuracy depends on OCR quality and document formatting
- Documents with complex layouts or low-resolution scans may yield poor results
- The regular expressions assume consistent document templates
- Non-Latin names may be partially lost during filename sanitization

## Troubleshooting

- **"poppler not found"**: Ensure Poppler's `bin` directory is in your system PATH
- **Poor OCR results**: Increase DPI in `pdf_to_text_whitelist` (default: 150)
- **Missing names**: Verify document format matches expected patterns; adjust regex if needed
- **Permission errors**: Ensure script has write access to the working directory

## License

This project is licensed under the MIT License. See the LICENSE file for details.