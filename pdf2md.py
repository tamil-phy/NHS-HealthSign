import logging
logging.basicConfig()

import argparse
from docling.document_converter import DocumentConverter

def convert(input_filepath, output_filepath):
    converter = DocumentConverter()
    try:
        result = converter.convert(input_filepath)
        with open(output_filepath, "w") as f:
            f.write(result.document.export_to_markdown())
    except:
        logging.exception(input_filepath)

    return output_filepath

def parse_args():
    parser = argparse.ArgumentParser(prog='PDF to Markdown Ocr')
    parser.add_argument('input', help='path to Report PDF file ')
    parser.add_argument('output', help='path to output markdown file')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    convert(args.input, args.output)
