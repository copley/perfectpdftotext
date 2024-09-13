import time
import psutil
import logging
from tqdm import tqdm
import pytesseract
from pdf2image import convert_from_bytes
import cv2
import numpy as np
import argparse
from multiprocessing import Pool
import PyPDF2
import os
import io

# Setting up logging for console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler("file3.log"), logging.StreamHandler()]
)

# Preprocess the image for better OCR accuracy
def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Use adaptive threshold for better results on varied images
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

# Function to process a single page
def process_page(page_image, language, page_num, total_pages):
    try:
        start_time = time.time()
        logging.info(f"Working on page {page_num + 1} of {total_pages}")
        print(f"Working on page {page_num + 1} of {total_pages}...")

        preprocessed_image = preprocess_image(page_image)
        text = pytesseract.image_to_string(preprocessed_image, lang=language)

        end_time = time.time()
        elapsed_time = end_time - start_time
        memory_info = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory in MB

        logging.info(f"Page {page_num + 1} processed in {elapsed_time:.2f} seconds. Memory usage: {memory_info:.2f} MB")
        print(f"Page {page_num + 1} processed in {elapsed_time:.2f} seconds. Memory usage: {memory_info:.2f} MB")
        return text
    except Exception as e:
        logging.error(f"Page {page_num + 1} failed to process. Error: {str(e)}")
        print(f"Page {page_num + 1} failed to process. Error: {str(e)}")
        return f"Error processing page {page_num + 1}\n"

# Function to read a PDF in-memory without writing chunks to disk
def read_pdf_in_memory(pdf_path, max_pages=100):
    logging.info(f"Reading {pdf_path} into memory...")
    print(f"Reading {pdf_path} into memory...")

    pdf_chunks = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)

        for i in range(0, total_pages, max_pages):
            pdf_writer = PyPDF2.PdfWriter()
            for j in range(i, min(i + max_pages, total_pages)):
                pdf_writer.add_page(pdf_reader.pages[j])

            # Write to a bytes buffer instead of disk
            pdf_buffer = io.BytesIO()
            pdf_writer.write(pdf_buffer)
            pdf_chunks.append(pdf_buffer)

            logging.info(f"Created in-memory chunk with pages {i + 1} to {min(i + max_pages, total_pages)}")
            print(f"Created in-memory chunk with pages {i + 1} to {min(i + max_pages, total_pages)}")

    return pdf_chunks

# Main function to process the entire PDF in chunks of 100 pages
def pdf_to_text(pdf_path, output_txt_path, language):
    try:
        pdf_chunks = read_pdf_in_memory(pdf_path)

        # Process each chunk
        for chunk_num, pdf_buffer in enumerate(pdf_chunks, 1):
            logging.info(f"Processing chunk {chunk_num}")
            print(f"Processing chunk {chunk_num}")

            # Convert chunk to images
            pdf_buffer.seek(0)  # Reset buffer pointer
            pages = convert_from_bytes(pdf_buffer.read())
            total_pages = len(pages)
            logging.info(f"Conversion of chunk {chunk_num} complete. {total_pages} pages detected.")
            print(f"Total pages in chunk {chunk_num}: {total_pages}")

            # Use os.cpu_count() for optimal pool size
            with Pool(processes=os.cpu_count()) as pool:
                with tqdm(total=total_pages, desc=f"Processing Chunk {chunk_num}", unit="page", dynamic_ncols=True) as progress_bar:
                    results = []
                    for i, page in enumerate(pages):
                        result = pool.apply_async(process_page, args=(page, language, i, total_pages), callback=lambda _: progress_bar.update(1))
                        results.append(result)
                    pool.close()
                    pool.join()

            # Write the extracted text to the output file
            with open(output_txt_path, 'a', encoding='utf-8') as output_file:
                for i, result in enumerate(results):
                    text = result.get()
                    output_file.write(f"--- Chunk {chunk_num}, Page {i + 1} ---\n")
                    output_file.write(text)
                    output_file.write("\n\n")

        logging.info(f"Text extraction complete! Results saved to {output_txt_path}")
        print(f"Text extraction complete! Results saved to {output_txt_path}")
    except Exception as e:
        logging.error(f"Failed to process the PDF. Error: {str(e)}")
        print(f"Error occurred: {str(e)}")

# Command-line interface for easier usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from a scanned PDF using OCR in chunks of 100 pages.")
    parser.add_argument('pdf_path', type=str, help="Path to the input PDF file")
    parser.add_argument('output_txt_path', type=str, help="Path to the output text file")
    parser.add_argument('--language', type=str, default='eng', help="Languages for OCR (default is English, e.g., 'eng+fra')")
    args = parser.parse_args()

    # Run the main function
    pdf_to_text(args.pdf_path, args.output_txt_path, args.language)

