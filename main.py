import os
import re
import glob
import logging
import time
import string
import concurrent.futures
from functools import partial
import unicodedata
from pdf2image import convert_from_path
import PyPDF2 as pdf
import pytesseract
from tqdm import tqdm
from functools import wraps
from typing import Callable
from pathlib import Path
import zipfile
from typing import Generator
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log", "a", "utf-8"), logging.StreamHandler()],
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

CERT_PATTERN: re.Pattern = re.compile(
    r"to\s+certify\s+that\s+"
    r"(?P<name>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s'-.]+?)"
    r"\s+has",
    flags=re.IGNORECASE | re.DOTALL,
)

REPT_PATTERN: re.Pattern = re.compile(
    r"Payer\s+(?P<name>.*?)\s+Amount", flags=re.IGNORECASE
)

MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)

WHITELIST: str = string.ascii_letters + "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ ._-',"

def brenchmark_func(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time: float = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"Function {func.__name__} took {end_time - start_time} seconds")
        return result

    return wrapper

def sanitize_filename(name: str) -> str:
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    invalid_chars = r'[<>:"/\\|?*]'
    name = re.sub(invalid_chars, '_', name)
    return name.strip() or "Unknown"

def pdf_to_text_whitelist(
    pdf_path: str, dpi: int = 150, lang: str = "eng", poppler_path: str = None
) -> Generator[str, None, None]:
    images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    tesseract_config = f'--psm 1 -c tessedit_char_whitelist="{WHITELIST}"'
    for img in images:
        text = pytesseract.image_to_string(img, lang=lang, config=tesseract_config)
        yield " ".join(text.split())


def extract_name_between(pages: list[str]) -> Generator[str | None, None, None]:
    for text in pages:
        m = REPT_PATTERN.search(text)
        yield m.group("name").strip() if m else None


def extract_names_from_pages(pages: list[str]) -> Generator[str | None, None, None]:
    for text in pages:
        m = CERT_PATTERN.search(text)
        yield m.group("name").strip() if m else None


def split_pdf_to_pages(
    file_path: str,
    output_folder: str,
    name_list: Generator[str | None, None, None],
    course_code: str,
    prefix: str,
) -> None:
    reader: pdf.PdfReader = pdf.PdfReader(file_path)
    name_list_iter = iter(name_list)
    for page_num, (page, name) in enumerate(zip(reader.pages, name_list_iter)):
        writer = pdf.PdfWriter()
        writer.add_page(page)
        safe_name = sanitize_filename(name or f"Page_{page_num}")
        output_filename = Path(output_folder) / f"{safe_name}_{course_code}_{prefix}.pdf"
        with open(output_filename, "wb") as out_file:
            writer.write(out_file)
        logger.info("Created: %s", output_filename)


def process_certificate(cert_path: str, course_code: str) -> None:
    logger.info("Processing certificate file: %s", cert_path)
    pages: Generator = pdf_to_text_whitelist(cert_path)
    names: Generator = extract_names_from_pages(pages)
    split_pdf_to_pages(
        cert_path, "Split_Certificates", names, course_code, "Certificate"
    )


def process_receipts(receipt_path: str, course_code: str) -> None:
    logger.info("Processing receipt file: %s", receipt_path)
    pages: Generator = pdf_to_text_whitelist(receipt_path)
    names: Generator = extract_name_between(pages)
    split_pdf_to_pages(receipt_path, "Split_Receipts", names, course_code, "Receipt")


def convert_to_zip(folder_path: str) -> None:
    with zipfile.ZipFile(f"{folder_path}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in Path(folder_path).iterdir():
            zipf.write(file_path, file_path.name)

    shutil.rmtree(folder_path)


def process_pdf(
    pdf_list: list[str], course_code: str, func: Callable, save_path: str
) -> None:
    process_func = partial(func, course_code=course_code)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(process_func, pdf_list), total=len(pdf_list), desc="Processing PDFs"))

    convert_to_zip(save_path)


@brenchmark_func
def main():
    certificates: list[str] = sorted(glob.glob("Certificates/*.pdf"))
    receipts: list[str] = sorted(glob.glob("Receipts/*.pdf"))


    if not certificates and not receipts:
        logger.info("No PDF files found in Certificates or Receipts folders.")
        return
    course_code: str = input("Enter course code: ").strip()
    if certificates:
        process_pdf(certificates, course_code, process_certificate, "Split_Certificates")
    if receipts:
        process_pdf(receipts, course_code, process_receipts, "Split_Receipts")


if __name__ == "__main__":
    os.makedirs("Split_Certificates", exist_ok=True)
    os.makedirs("Split_Receipts", exist_ok=True)
    main()