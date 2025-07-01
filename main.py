import os
import re
import glob
import logging
import time
import string
import concurrent.futures
from functools import partial
from pdf2image import convert_from_path
import PyPDF2 as pdf
import pytesseract
from tqdm import tqdm
from functools import wraps
from typing import Callable
from pathlib import Path
import zipfile
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

CERT_PATTERN = re.compile(
    r"to\s+certify\s+that\s+"
    r"(?P<name>[A-Za-z][A-Za-z\s'-]+?)"
    r"\s+has",
    flags=re.IGNORECASE,
)

MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)


def brenchmark_func(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time: float = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time} seconds")
        return result

    return wrapper


@brenchmark_func
def pdf_to_text_whitelist(
    pdf_path: str, dpi: int = 150, lang: str = "eng", poppler_path: str = None
) -> list[str]:
    images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    whitelist: str = string.ascii_letters + " ._-"
    tesseract_config = f'--psm 1 -c tessedit_char_whitelist="{whitelist}"'
    pages: list[str] = []
    for img in images:
        text = pytesseract.image_to_string(img, lang=lang, config=tesseract_config)
        text = " ".join(text.split())
        pages.append(text)
    return pages


def extract_name_between(pages: list[str]) -> list[str | None]:
    pattern: re.Pattern = re.compile(
        r"Payer\s+(?P<name>.*?)\s+Amount", flags=re.IGNORECASE
    )
    names: list[str | None] = []
    for text in pages:
        m = pattern.search(text)
        if m:
            name = m.group("name").strip()
            names.append(name)
        else:
            names.append(None)
    return names


def extract_names_from_pages(pages: list[str]) -> list[str | None]:
    names: list[str | None] = []
    for text in pages:
        m = CERT_PATTERN.search(text)
        if m:
            name = m.group("name").strip()
            names.append(name)
        else:
            names.append(None)
    return names


def split_pdf_to_pages(
    file_path: str,
    output_folder: str,
    name_list: list[str | None],
    course_code: str,
    prefix: str,
) -> list[str]:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        os.makedirs(output_folder, exist_ok=True)
        reader: pdf.PdfReader = pdf.PdfReader(file_path)

        if len(name_list) != len(reader.pages):
            raise ValueError(
                f"Length of name_list ({len(name_list)}) does not match number of PDF pages ({len(reader.pages)})"
            )

        output_files: list[str] = []
        for page_num, name in enumerate(name_list):
            try:
                writer = pdf.PdfWriter()
                writer.add_page(reader.pages[page_num])
                safe_name = (name or "Unknown").strip()
                output_filename = os.path.join(
                    output_folder, f"{safe_name}_{course_code}_{prefix}.pdf"
                )
                with open(output_filename, "wb") as out_file:
                    writer.write(out_file)
                output_files.append(output_filename)
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {str(e)}")
                continue
        return output_files
    except Exception as e:
        logger.error(f"Error in split_pdf_to_pages: {str(e)}")
        raise


def process_certificate(cert_path: str, course_code: str) -> None:
    logger.info(f"Processing certificate file: {cert_path}")
    pages: list[str] = pdf_to_text_whitelist(cert_path)
    names: list[str | None] = extract_names_from_pages(pages)
    logger.info(f"Extracted names: {names}")
    split_pdf_to_pages(
        cert_path, "Split_Certificates", names, course_code, "Certificate"
    )


def process_receipts(receipt_path: str, course_code: str) -> None:
    logger.info(f"Processing receipt file: {receipt_path}")
    pages: list[str] = pdf_to_text_whitelist(receipt_path)
    names: list[str | None] = extract_name_between(pages)
    logger.info(f"Extracted names: {names}")
    split_pdf_to_pages(receipt_path, "Split_Receipts", names, course_code, "Receipt")


def convert_to_zip(folder_path: str) -> None:
    """lazy evaluation with generator"""
    with zipfile.ZipFile(f"{folder_path}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in Path(folder_path).iterdir():
            zipf.write(file_path, file_path.name)

    # Delete original folder (after ZIP is confirmed written)
    shutil.rmtree(folder_path)


def process_cert(cert, course_code: str) -> None:
    if not cert:
        logger.info("No certificate PDFs found.")
        return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        process_func = partial(process_certificate, course_code=course_code)
        list(tqdm(executor.map(process_func, cert), total=len(cert)))

    convert_to_zip("Split_Certificates")


def process_receipt(receipt, course_code: str) -> None:
    if not receipt:
        logger.info("No receipt PDFs found.")
        return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        process_func = partial(process_receipts, course_code=course_code)
        list(tqdm(executor.map(process_func, receipt), total=len(receipt)))

    convert_to_zip("Split_Receipts")


@brenchmark_func
def main():
    course_code: str = input("Enter course code: ").strip()
    certificates: list[str] = sorted(glob.glob("Certificates/*.pdf"))
    receipts: list[str] = sorted(glob.glob("Receipts/*.pdf"))

    process_cert(certificates, course_code)
    process_receipt(receipts, course_code)


if __name__ == "__main__":
    main()
