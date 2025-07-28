# Challenge 1A

## Approach

- Trained a Random Forest Classifier that classifies Headings from Non-Headings over 50 PDFs
- Used a rule-based model to level the collected Headings

### Random Forest Classifier

| Class | Context     |
| ----- | ----------- |
| 0     | Non-Heading |
| 1     | Heading     |

| Class     | Precision | Recall | F1 - Score |
| --------- | --------- | ------ | ---------- |
| 0         | 0.99      | 0.99   | 0.99       |
| 1         | 0.90      | 0.92   | 0.91       |
| macro-avg | 0.95      | 0.95   | 0.95       |
| accuracy  |           |        | 0.99       |

### Rule-based Model

- Rule 1: A Heading at a higher place should be a heading of higher level or same.
- Rule 2: A Heading with higher font size should be a heading of higher level.
- Rule 3: A Heading with higher font weight should be a heading of higher level.
- Rule 4: A Heading with more prominent case should be a heading of higher level (Like a Title Case headings should be a heading of higher level when compared to a heading of Upper Case).

### Title Selection

- Selected the first contiguous section of text with the highest font size in the first page and stored each line in a list named `title_list`.
- Ran a Cosine Similarity with the word `"document title"` with every subarray of the `title_list`.
- Assuming that the PDF doesn't have a title if the Cosine Similarity of all subarray is less than `0.01`.

## How to run Challenge 1A

- In the directory `./input/` store all the PDFs that we need the outline of.

- Run the following docker command to build a docker image

```docker
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier
```
- After building the image run the following docker command to execute the built image (Note that this will also run Challenge 1B with the given testcases)

```docker
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none mysolutionname:somerandomidentifier
```

- This runs `1a.py` that is the python script to extract outlines of all the PDFs inside `./input`
- The resulting outlines are stored in a json format in the directory `./output/`

### JSON Format

```json
{
    "title": "Title of the PDF",
    "outline": [
		{
            "level": "H1",
            "text": "H1 Level Heading",
            "page": 1
        },
		{
            "level": "H2",
            "text": "H2 Level Heading",
            "page": 2
        },
		{
            "level": "H3",
            "text": "H3 Level Heading",
            "page": 3
        }
	]
}
```

# Challenge 1B

## Approach

- Getting the outline using steps used in Challenge 1A.
- Takes persona, document list and job from a file named `input.json`.
- Structure the query as: `"query: {persona} is asking, {job}"`
- Calculating Cosine Simialrity with all the headings extracted from all the PDFs and picking the top 5 headings (or less if there are less that 5 headings) with the highest Cosine Similarity.
- Calculating Cosine Similarity with all paragraphs with their headings (the format is `"{Heading}: {Paragraph}"`) and picking top 5 paragraphs (or less if there are less that 5 paragraphs) with the highest Cosine Similarity.
- Showing the most relevant headings in the `extracted_sections` part of the `output.json`.
- Showing the most relevant paragraphs in the `subsection_analysis` part of the `output.json`.

### Input JSON Format
```json
{
    "challenge_info": {
        "challenge_id": "round_1b_003",
        "test_case_name": "test case name",
        "description": "description of test case"
    },
    "documents": [
        {
            "filename": "File 1.pdf",
            "title": "Title of File 1"
        },
        {
            "filename": "File 2.pdf",
            "title": "Title of File 2"
        },
        {
            "filename": "File 3.pdf",
            "title": "Title of File 3"
        }
    ],
    "persona": {
        "role": "role of the persona"
    },
    "job_to_be_done": {
        "task": "task needed by the persona"
    }
}
```

### Output JSON Format

```json
{
    "metadata": {
        "input_documents": [
            "File 1.pdf",
			"File 2.pdf"
        ],
        "persona": "role of the persona",
        "job_to_be_done": "task needed by the persona",
        "processing_timestamp": "Timestamp in ISO format like 2025-07-10T15:31:22.632389"
    },
    "extracted_sections": [
        {
            "document": "File 1.pdf",
            "section_title": "Relevant Heading",
            "importance_rank": 1,
            "page_number": 1
        },
        {
            "document": "File 2.pdf",
            "section_title": "Relevant Heading",
            "importance_rank": 2,
            "page_number": 3
        }
    ],
    "subsection_analysis": [
        {
            "document": "File 1.pdf",
            "refined_text": "Relevant Paragraph",
            "page_number": 1
        },
		{
            "document": "File 1.pdf",
            "refined_text": "Relevant Paragraph",
            "page_number": 3
        },
		{
            "document": "File 2.pdf",
            "refined_text": "Relevant Paragraph",
            "page_number": 2
        }
    ]
}
```

## How to run Challenge 1B

- The Dockerfile contains instructions to run both Challenge 1A and Challenge 1B
- Docker commands are the following (the same used in Challenge 1A)

```docker
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none mysolutionname:somerandomidentifier
```

- This runs `1b.py` that is the python script to run the given testcases of Challenge 1B stored in `./challenge_1b/Collection_1/PDFs/` and `./challenge_1b/Collection_2/PDFs`. The `input.json` file of each testcase is respectively stored in `./challenge_1b/Collection_1/` and `./challenge_1b/Collection_2/`.
- The `output.json` is file of each testcase is respectively generated in `./challenge_1b/{challenge_id}.json`.