import os
import json
import re
import logging
import datetime
import hashlib
import openai
from fuzzywuzzy import fuzz
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

READCODE_PATH = "red_flag_read_codes_subset.json"

def get_api_key(api):
    with open("secrets.json") as f:
        secrets = json.load(f)
    return secrets['API_KEY'][api]

os.environ['OPENAI_API_KEY'] = get_api_key('OPENAI')
LLM_MODEL_NAME = "gpt-4o-mini-2024-07-18"
OPENAI_LLM = openai.OpenAI()

def talk_to_llm(messages, text):
    response = OPENAI_LLM.responses.create(
        model=LLM_MODEL_NAME,
        input=messages,
        text=text,
        temperature=0
    )
    return json.loads(response.output_text)

def parse_markdown(file_path):
    lines = []
    current_date = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("## "):
                current_date = line[3:].strip()
            else:
                if current_date:
                    lines.append(f'[{current_date}]: {line}')
                else:
                    lines.append(line)
    return lines

def read_markdown_lines(file_path):
    return parse_markdown(file_path)

def extract_vital_by_alias(vital_test_name, vital_details, lines):
    alias_list = vital_details.get("aliases", [])
    unit_list = vital_details.get("units", [])
    matching_lines = []
    for line in lines:
        for alias in alias_list:
            if alias.lower() in line.lower():
                matching_lines.append(line)
                break
    if not matching_lines:
        return None
    text_subset = "\n".join(matching_lines)
    prompt = f"""
You are an assistant tasked with extracting the vital measurement "{vital_test_name}" from the text below.
The text may include multiple measurements with dates. If a date is present it will be in the format "[date]:" at the beginning of the line.
Extract for each occurrence:
  - date (if available),
  - the measurement value,
  - the measurement unit (allowed units: {', '.join(unit_list)}),
  - and the full matching line as context.
Return only JSON in the following structure (without extra fields):

{{
  "vital_name": "{vital_test_name}",
  "measurements": [
    {{
      "date": "date string if available",
      "value": "measurement value as string",
      "unit": "measurement unit",
      "context": "the full matching line"
    }}
  ]
}}
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text_subset}
    ]
    schema = {
        "type": "object",
        "properties": {
            "vital_name": {"type": "string"},
            "measurements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "value": {"type": "string"},
                        "unit": {"type": "string"},
                        "context": {"type": "string"}
                    },
                    "required": ["date", "value", "unit", "context"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["vital_name", "measurements"],
        "additionalProperties": False
    }
    response = talk_to_llm(
        messages,
        {"format": {
            "type": "json_schema",
            "name": "vital_extraction",
            "schema": schema,
            "strict": True
        }}
    )
    return response

def extract_all_vitals(file_path, vital_lookup_file):
    lines = read_markdown_lines(file_path)
    with open(vital_lookup_file, 'r') as f:
        vital_lookup = json.load(f)
    all_results = []
    for category, cat_data in tqdm(vital_lookup.items(), desc="Processing Vital Categories"):
        tests = cat_data.get("tests", {})
        for test_name, test_details in tqdm(tests.items(), desc="Processing Tests", leave=False):
            logging.info(f"Extracting {test_name} ...")
            result = extract_vital_by_alias(test_name, test_details, lines)
            if result:
                all_results.append(result)
    return {"vitals": all_results}

def match_read_code_in_line(line, code, description):
    if len(line.strip()) < 10:
        return False
    if code in line:
        return True
    if fuzz.partial_ratio(description.lower(), line.lower()) > 85:
        return True
    return False

def enrich_lines_with_read_codes(parsed_lines, readcode_dict):
    enriched = []
    for line in tqdm(parsed_lines, desc="Processing Read Codes"):
        date = re.match(r'\[(.*)\]:', line).group(1)
        clean_line = line.replace(f'[{date}]:', '')
        line_data = {
            "date": date,
            "line": clean_line,
            "matched_category": None,
            "matched_group": None,
            "matched_code": None,
            "code_description": None
        }
        for category, details in readcode_dict.items():
            for group in details.get("code_groups", []):
                for code, desc in group["codes"].items():
                    if match_read_code_in_line(clean_line, code, desc):
                        line_data["matched_category"] = category
                        line_data["matched_group"] = group["group"]
                        line_data["matched_code"] = code
                        line_data["code_description"] = desc
                        enriched.append(line_data)
                        break
                if line_data["matched_code"]:
                    break
            if line_data["matched_code"]:
                break
    return enriched

def generate_summary_chronology_chunked(filepath, output_dir):
    PROMPT = """
    You are summarizing the major clinical events from the NHS report.
    Return the events in descending order. List only abnormal events that would be relevant to an underwriting officer,
    but do not mention anything about insurance or underwriting in your response.
    Avoid duplicate events; when you find multiple measurements for the same vital, use the latest one.
    Provide a detailed one-paragraph description including diagnosis and numerical test values.
    Return only JSON in the following structure without extra fields.
    """
    events_schema = {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "significance": {"type": "number"}
                    },
                    "required": ["date", "title", "description", "significance"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["events"],
        "additionalProperties": False
    }
    lines = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                lines.append(line.strip())
    chunks = chunk_lines(lines)
    chunk_summaries = []
    for i, chunk in enumerate(tqdm(chunks, desc="Timeline Chunks")):
        prompt_text = "Chronology events:\n" + "\n".join(chunk)
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": prompt_text}
        ]
        write_intermediate_results(
            output_dir,
            f'timeline-output-chunk-{i}.txt',
            '\n'.join(chunk),
            f'timeline entries for chunk-{i}'
        )
        logging.info(f"Processing timeline chunk {i+1}/{len(chunks)} with {len(chunk)} lines...")
        response = talk_to_llm(
            messages=messages,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "clinical_events",
                    "schema": events_schema,
                    "strict": True
                }
            }
        )
        chunk_summary = response
        write_intermediate_results(
            output_dir,
            f'timeline-output-chunk-{i}-output.json',
            json.dumps(chunk_summary),
            f'timeline entries for chunk-{i} output'
        )
        chunk_summaries.append(chunk_summary)
    if len(chunk_summaries) > 1:
        combined_text = ""
        for summary in chunk_summaries:
            for event in summary.get("events", []):
                if event['significance'] > 3:
                    combined_text += f"{event['date']} - {event['title']} - {event['description']}\n"
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": combined_text}
        ]
        final_response = talk_to_llm(
            messages=messages,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "clinical_events",
                    "schema": events_schema,
                    "strict": True
                }
            }
        )
        final_summary = final_response
    else:
        final_summary = chunk_summaries[0]
    write_intermediate_results(
        output_dir,
        f'timeline-output-output.json',
        json.dumps(final_summary),
        f'timeline output'
    )
    return final_summary

def chunk_lines(lines, max_tokens=1000):
    chunks = []
    current_chunk = []
    current_tokens = 0
    for line in lines:
        line_tokens = len(line.split())
        if current_tokens + line_tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = [line]
            current_tokens = line_tokens
        else:
            current_chunk.append(line)
            current_tokens += line_tokens
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def generate_summary_major_disease_llm(read_code_data, output_dir):
    PROMPT = """
    You are summarizing the major diseases from the NHS report using the provided read code extractions.
    Return the diseases in descending order of relevance.
    Provide a one-paragraph description for each disease that includes the read code, description, and date.
    Return only JSON in the following structure without extra fields.
    """
    schema = {
        "type": "object",
        "properties": {
            "diseases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "read_code": {"type": "string"},
                        "description": {"type": "string"},
                        "severity": {"type": "number"}
                    },
                    "required": ["date", "read_code", "description", "severity"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["diseases"],
        "additionalProperties": False
    }
    combined_text = ""
    for entry in read_code_data:
        combined_text += f"{entry['date']} - {entry['matched_code']} - {entry['code_description']}\n"
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": combined_text}
    ]
    response = talk_to_llm(
        messages=messages,
        text={
            "format": {
                "type": "json_schema",
                "name": "major_disease_summary",
                "schema": schema,
                "strict": True
            }
        }
    )
    write_intermediate_results(
        output_dir,
        "major-disease-summary.json",
        json.dumps(response),
        "major disease summary output"
    )
    return response

def make_session_directory(filename):
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M")
    h = hashlib.new('sha256')
    h.update(b"{filename}")
    tempdir = h.hexdigest()
    tempdir = f"tmp/{timestamp}--{tempdir[-8:]}"
    os.makedirs(tempdir, exist_ok=True)
    return tempdir

def write_intermediate_results(output_dir, filepath, text, log_message):
    filepath = filepath.lower().replace(' ', '-')
    filepath = f'{output_dir}/{filepath}'
    logging.info(f'writing intermediate results {log_message} to {filepath}')
    with open(filepath, 'w') as f:
        f.write(text)

def parse_regexp(text):
    regexps = {
        "coded-entry": r"- Coded entry - (?P<entry>.*)$",
        "organization": r"(?P<entry>([\w\s\(\)]+) - ([\w\s\-]*)(| \((.*)\)))",
        "vaccination": r"- Vaccination - (?P<entry>.*)",
        "vaccination-consent": r"- Vaccination consent - Vaccination Consent: (?P<entry>.*)",
        "medication": r"- Medication - (?P<entry>.*)",
        "medication-template": r"- Medication template - (?P<entry>.*)",
        "date": r"## (?P<entry>\d+\s+\w+\s+\d+)",
        "test-result": r"- Test result - (?P<entry>.*)$",
        "test-request": r"- Test request - (?P<entry>.*)$",
        "referral": r"- Referral out - (?P<entry>.*)"
    }
    if not text.strip():
        return None, None
    for tag, pattern in regexps.items():
        match = re.match(pattern, text)
        if match:
            return tag, match.groupdict()
    else:
        return None, f'No matches:\n  {text}'

def parse_markdown(file_path):
    lines = []
    current_date = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            tag, groupdict = parse_regexp(line)
            if tag is None and groupdict is None:
                continue
            if tag is None:
                lines.append(f'[UNPARSED]: {line}')
            elif tag == 'date':
                current_date = groupdict['entry']
            else:
                try:
                    lines.append(f'[{current_date}]: {line}')
                except Exception as e:
                    logging.exception(line)
    return lines

def process(input_path, session_dir=None):
    session_dir = session_dir or make_session_directory("output_amma.md")
    os.makedirs(session_dir, exist_ok=True)
    if re.search('[pP][dD][fF]', os.path.splitext(args.input)[1]):
        ocr_filepath = f"{args.output_dir}/report.md"
        pdf2md.convert(args.input, ocr_filepath)
        args.input = ocr_filepath
    input_path = args.input
    parsed_lines = parse_markdown(input_path)
    OUTPUT_PATH = {
        "code_matches": f"{session_dir}/code_matches.json",
        "vital_matches": f"{session_dir}/vital_matches.json",
        "timeline": f"{session_dir}/timeline.json"
    }
    logging.info('Extracting major diseases (read codes) ...')
    with open(READCODE_PATH, 'r') as f:
        READCODE_DICT = json.load(f)
    code_matches = enrich_lines_with_read_codes(parsed_lines, READCODE_DICT)
    with open(OUTPUT_PATH["code_matches"], "w") as f:
        json.dump(code_matches, f, indent=2, ensure_ascii=False)
    logging.info('Extracting vitals using LLM ...')
    vitals_extracted = extract_all_vitals(input_path, args.lookup)
    with open(OUTPUT_PATH["vital_matches"], "w") as f:
        json.dump(vitals_extracted, f, indent=2, ensure_ascii=False)
    logging.info('Generating timeline summary ...')
    timeline_summary = generate_summary_chronology_chunked(input_path, session_dir)
    with open(OUTPUT_PATH["timeline"], "w") as f:
        json.dump(timeline_summary, f, indent=2, ensure_ascii=False)
    logging.info('Generating major disease summary using LLM ...')
    major_disease_summary = generate_summary_major_disease_llm(code_matches, session_dir)
    with open(f"{session_dir}/major_disease_summary.json", "w") as f:
        json.dump(major_disease_summary, f, indent=2, ensure_ascii=False)
    return OUTPUT_PATH

if __name__ == "__main__":
    class Args:
        input = "your_input_file.md"
        lookup = "INTERESTED_VITALS.json"
        output_dir = "output_trial"
    args = Args()
    process(args.input, args.output_dir)