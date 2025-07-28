import pymupdf
import os
import regex
from joblib import load
import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util
import torch
from datetime import datetime


class PDFDataExtractor:
    flags = pymupdf.TEXTFLAGS_DICT & ~ pymupdf.TEXT_PRESERVE_IMAGES
    lowercase_words = ["a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet", "at", "by", "for", "from", "in", "into", "like", "near", "of", "off", "on", "onto", "out", "over", "past", "to", "up", "with", "as", "if", "than", "that", "though", "when", "then", "which", "while"]
    headings = dict()
    data = dict()
    paragraphs = []
    file_list = []
    
    def __init__(self, folder_path : str = './app/input', model_path : str = "./models/RandomForest.joblib", output_path : str ='./app/output'):
        self.LLM_model_path = r"./llm_model"
        self.model =SentenceTransformer(self.LLM_model_path)
        self.pipeline = load(model_path)
        self.folder_path = folder_path
        self.file_list = [f for f in os.listdir(self.folder_path) if f.endswith('.pdf') and os.path.isfile(os.path.join(self.folder_path, f))]
        self.output_path = output_path

    def buildSizePercentileDict(self, a : dict):
        sorted_items = sorted(a.items())
        total = sum(a.values())
        
        percentile_dict = {}
        cumulative = 0
        
        for key, value in sorted_items:
            percentile_dict[key] = (cumulative / total) * 100
            cumulative += value
            
        return percentile_dict, total
    
    def sanitizeText(self, text : str):
        text = text.strip()
        text = regex.sub(r'\b(?!a\b)([A-Za-z])\s+', r'\1', text)
        text = regex.sub(r'\s+', ' ', text)
        return text
    
    def extractTextFeatures(self, text : str):
        text = self.sanitizeText(text)
        words = regex.findall(r'\b\p{L}+\b', text)
        word_count = len(words)
        
        
        digit_count = len(regex.findall(r'\b\(?\p{N}+\)?\b', text))
        roman_numerals = 1 if regex.search(r'\b[IVXLCMivxlcm]+\b', text) else 0
        upper_words = sum(1 for w in words if w.istitle())
        lower_words = word_count - upper_words
        total_titles = sum(1 for w in words if w.istitle() and w not in self.lowercase_words)
        word_count2 = sum(1 for w in words if w not in self.lowercase_words)
        caps_ratio = sum(1 for c in text.replace(' ', '') if c.isupper())/len(text.replace(' ', ''))
        
        # 0 for more caps ratio, 1 for more titles, 2 for more upper case, 3 for more lower case
        case = 0
        if caps_ratio < 0.95:
            if total_titles != word_count2:
                case = int(lower_words > upper_words) + 2
            else: case = 1
        
        # print(words, upper_words, total_titles, lower_words)
        
            
        return word_count, digit_count, roman_numerals, case, caps_ratio
    
    def extractBboxFeatures(self, coords : tuple, page_width : float):
        x0, y0, x1, y1 = coords
        
        bbox_width, bbox_height = x1 - x0, y1 - y0
        x_center = abs(((x0 + x1) / 2) - page_width)
        indent = x0/page_width
        indent = float("{:.2f}".format(indent))
        box_aspect = bbox_height/bbox_width if bbox_width else 0
        
        return bbox_width, bbox_height, x_center, indent, box_aspect
    
    def extractPDF(self, file_path : str, training_label : bool = False):
        if not os.path.exists(file_path):
            return -1
        doc = pymupdf.open(file_path)
        file_name = os.path.basename(file_path)
        pages = len(doc)
        rows = []
        font_dict = dict()
        avg_font_size = 0
        
        for i in range(pages):
            page = doc[i]
            tables_list = []
            try:
                blocks = page.get_text("dict", flags=self.flags)["blocks"]
            except Exception as e:
                print(f"Warning: error on {file_path} page {i+1}: {e}")
                continue
            
                
            for block in blocks:
                for line in block['lines']:
                    for span in line['spans']:
                        text = span['text'].strip()
                        if not text: continue
                        text_len = len(text)
                        font_size = round(span['size'])
                        avg_font_size += font_size*text_len
                        font_dict[font_size] = font_dict.get(font_size, 0) + text_len
                    
        font_dict, no_lines = self.buildSizePercentileDict(font_dict)
        avg_font_size = round(avg_font_size/no_lines)
        
        for i in range(pages):
            page = doc[i]
            page_width = page.bound()[2]
            try:
                blocks = page.get_text("dict", flags=self.flags)["blocks"]
            except Exception as e:
                print(f"Warning: error on {file_path} page {i+1}: {e}")
                continue        

            no_of_blocks = blocks[-1]['number'] + 1 if blocks else 0
            
            for block in blocks:
                pos = block['number']
                curr_line_count = len(block['lines'])
                for line_pos, line in enumerate(block['lines']):
                    space_above = space_below = 0
                    
                    if line_pos != 0:
                        space_above = line['bbox'][1] - block['lines'][line_pos - 1]['bbox'][3]
                    elif pos != 0:
                        space_above = line['bbox'][1] - blocks[pos - 1]['lines'][-1]['bbox'][3]
                    
                    if line_pos != curr_line_count-1:
                        space_below = block['lines'][line_pos + 1]['bbox'][1] - line['bbox'][3]
                    elif pos != no_of_blocks-1:
                        space_below = blocks[pos + 1]['lines'][0]['bbox'][1] - line['bbox'][3]

                    
                    space_above = max(space_above, 0)
                    space_below = max(space_below, 0)
                    
                    # print(pos, space_above if space_above>0 else 0, space_below if space_below>0 else 0)
                    line_text = ""
                    total_spans = 0
                    coords = line['bbox']
                    
                    bbox_width, bbox_height, x_center, indent, box_aspect = self.extractBboxFeatures(coords, page_width)
                    # pprint(line)
                    is_bold = 1
                    is_italic = 1
                    font_size = 0
                    
                    line_text = ' '.join([span['text'] for span in line["spans"]])
                    line_text = line_text.strip()
                    line_text = regex.sub(r'\b(?!a\b)([A-Za-z])\s+', r'\1', line_text)
                    line_text = regex.sub(r'\s+', ' ', line_text)
                    total_spans = sum(1 for span in line['spans'] if span['text'].strip())
                    if not line_text: continue
                    
                    for span in line['spans']:
                        text = span['text'].strip()
                        if not text: continue
                        is_bold = is_bold & bool(span['flags'] & 16)
                        is_italic = is_italic & bool(span['flags'] & 2)
                        if not font_size:
                            font_size = round(span['size'])
                    
                    
                    word_count, digit_count, roman_numerals, case, caps_ratio = self.extractTextFeatures(line_text)
                    # if not line_text: continue
                    curr_dict = {
                        "file_name": file_name,
                        "page": i+1,
                        "text": line_text,
                        "coord": (coords[3], coords[0]), #(y1,x0)
                        "font_size": font_size,
                        # "font_threshold": int(font_size > avg_font_size),
                        "font_percentile": font_dict[font_size],
                        "is_bold": is_bold,
                        "is_italic": is_italic,
                        "word_count": word_count,
                        "digit_count": digit_count,
                        "roman_numerals": roman_numerals,
                        "case": case,
                        "caps_ratio": caps_ratio,
                        "bbox_width": bbox_width,
                        "bbox_height": bbox_height,
                        "x_center": x_center,
                        "indent": indent,
                        "box_aspect": box_aspect,
                        # "total_spans": total_spans,
                        "space_above": space_above,
                        "space_below": space_below,
                        "total_space": space_above + space_below,
                    }
                    if training_label: curr_dict["label"] = 0
                    rows.append(curr_dict)
            
        return rows
    
    def getTitle(self, data : pd.DataFrame):
        first_page = data[data.page == 1]
        max_font_size = first_page['font_size'].max()
        title_list = []
        title_y1 = (-1,-1)
        find = False
        for row in first_page.itertuples():
            if row.font_size == max_font_size:
                find = True
                title_list.append((row.Index, row.text.strip(), row.coord))
                title_y1 = max(title_y1, row.coord)
            elif find:
                break
        lines = [x[1] for x in title_list]
    
        positive_ref = "document title"
        # negative_ref = "heading"

        pos_emb = self.model.encode(positive_ref, convert_to_tensor=True)
        # neg_emb = self.model.encode(negative_ref, convert_to_tensor=True)

        best_score = float("-inf")
        best_segment = ""

        for i in range(len(lines)):
            for j in range(i + 1, len(lines) + 1):
                segment = " ".join(lines[i:j])
                segment_emb = self.model.encode(segment, convert_to_tensor=True)

                score_pos = util.cos_sim(segment_emb, pos_emb)
                # score_neg = util.cos_sim(segment_emb, neg_emb)

                # score = score_pos - score_neg
                score = score_pos

                if score > best_score:
                    best_score = score
                    best_segment = segment
                    title_y1 = title_list[j-1][2]

        if best_score<0.01:
            best_segment = ''
            title_y1 = (-1, -1)
        
        return best_segment.strip(), title_y1, title_list
            
    def levelHeadingsWithSignature(self, df: pd.DataFrame):
        features = ["page", "text", "font_size", "is_bold", "is_italic", "indent", "case", "caps_ratio"]
        signature_features = ["font_size", "is_bold", "case"]
        
        X = df[features]
        X["heading_level"] = 1
        X["debug"] = -1
        
        signature = tuple(X.iloc[0][signature_features])
        signature_to_level = {signature:1}
        index_list = X.index.tolist()
        
        for i, row in enumerate(X.itertuples()):
            if i == 0:
                continue
            signature = tuple(getattr(row, feature) for feature in signature_features)
            if signature in signature_to_level:
                X.loc[index_list[i], "heading_level"] = signature_to_level[signature]
            else:
                last = X.loc[index_list[i-1]]
                curr_heading_level = last.heading_level
                # print(i, last.text)
                lh = last.heading_level
                if row.font_size > last.font_size:
                    curr_heading_level = max(lh - 1, 1)
                    X.loc[index_list[i], "debug"] = 0
                elif row.font_size < last.font_size:
                    curr_heading_level = lh + 1
                    X.loc[index_list[i], "debug"] = 0
                else:
                    if row.is_bold == 0 and last.is_bold == 1:
                        curr_heading_level = lh + 1
                        X.loc[index_list[i], "debug"] = 1
                    elif row.is_bold == 1 and last.is_bold == 0:
                        curr_heading_level = max(lh - 1, 1)
                        X.loc[index_list[i], "debug"] = 1
                    else:
                        if row.case == 0 and row.case < last.case:
                            curr_heading_level = max(lh - 1, 1)
                            X.loc[index_list[i], "debug"] = 3
                        elif last.case == 0 and row.case > last.case:
                            curr_heading_level = lh + 1
                            X.loc[index_list[i], "debug"] = 3
                        # if abs(row.indent - last.indent) < indent_tolerance:
                        #     if row.case == 0 and row.case < last.case:
                        #         curr_heading_level = max(lh - 1, 1)
                        #         X.loc[index_list[i], "debug"] = 3
                        #     elif last.case == 0 and row.case > last.case:
                        #         curr_heading_level = lh + 1
                        #         X.loc[index_list[i], "debug"] = 3
                        # else:
                        #     if row.indent  > last.indent - indent_tolerance:
                        #         curr_heading_level = lh + 1
                        #         X.loc[index_list[i], "debug"] = 2
                        #     elif row.indent < last.indent + indent_tolerance:
                        #         curr_heading_level = max(lh - 1, 1)
                        #         X.loc[index_list[i], "debug"] = 2
                            
                
                if curr_heading_level > 3:
                    df.loc[index_list[i], "label"] = 0
                    continue
                
                signature_to_level[signature] = curr_heading_level
                X.loc[index_list[i], "heading_level"] = curr_heading_level
        
        return X

    def levelHeadings(self, df: pd.DataFrame):
        features = ["page", "text", "font_size", "is_bold", "is_italic", "indent", "case", "caps_ratio"]
        # signature_features = ["font_size", "is_bold", "case"]
        
        # X = df[features]
        df["heading_level"] = 1
        # df["debug"] = -1
        
        # signature = tuple(df.iloc[0][signature_features])
        # signature_to_level = {signature:1}
        index_list = df.index.tolist()
        
        for i, row in enumerate(df.itertuples()):
            if i == 0:
                continue
            last = df.loc[index_list[i-1]]
            curr_heading_level = last.heading_level
            # print(i, last.text)
            lh = last.heading_level
            if row.font_size > last.font_size:
                curr_heading_level = max(lh - 1, 1)
                # df.loc[index_list[i], "debug"] = 0
            elif row.font_size < last.font_size:
                curr_heading_level = lh + 1
                # df.loc[index_list[i], "debug"] = 0
            else:
                if row.is_bold == 0 and last.is_bold == 1:
                    curr_heading_level = lh + 1
                    # df.loc[index_list[i], "debug"] = 1
                elif row.is_bold == 1 and last.is_bold == 0:
                    curr_heading_level = max(lh - 1, 1)
                    # df.loc[index_list[i], "debug"] = 1
                else:
                    if row.case == 0 and row.case < last.case:
                        curr_heading_level = max(lh - 1, 1)
                        # df.loc[index_list[i], "debug"] = 3
                    elif last.case == 0 and row.case > last.case:
                        curr_heading_level = lh + 1
                        # df.loc[index_list[i], "debug"] = 3
                    # if abs(row.indent - last.indent) < indent_tolerance:
                    #     if row.case == 0 and row.case < last.case:
                    #         curr_heading_level = max(lh - 1, 1)
                    #         df.loc[index_list[i], "debug"] = 3
                    #     elif last.case == 0 and row.case > last.case:
                    #         curr_heading_level = lh + 1
                    #         df.loc[index_list[i], "debug"] = 3
                    # else:
                    #     if row.indent  > last.indent - indent_tolerance:
                    #         curr_heading_level = lh + 1
                    #         df.loc[index_list[i], "debug"] = 2
                    #     elif row.indent < last.indent + indent_tolerance:
                    #         curr_heading_level = max(lh - 1, 1)
                    #         df.loc[index_list[i], "debug"] = 2
                        
            
            if curr_heading_level > 3:
                df.loc[index_list[i], "label"] = 0
                continue
            
            df.loc[index_list[i], "heading_level"] = curr_heading_level
        
        return df
    
    def getParagraph(self, df : pd.DataFrame, title_list : list, headings : pd.DataFrame):
        start = title_list[-1][0] + 1 if title_list else 0
        curr_para = []
        end = headings.index[0] if not headings.empty else df.index[-1]+1
        
        for i in range(start,end):
            curr_para.append(df.loc[i, "text"])
        para = [(1, df.loc[start, "file_name"], " ".join(curr_para))] if curr_para else []
        
        heading_indices = headings.index.to_list() + [df.index[-1]+1]
        
        for i in range(len(heading_indices)-1):
            start = heading_indices[i]
            end = heading_indices[i + 1]
            texts = df.loc[start:end - 1, "text"].tolist()
            if len(texts)>1:
                para_text = texts[0] + ': ' + " ".join(texts[1:])
                para.append((df.loc[start, "page"], df.loc[start, "file_name"], para_text))
        
        return para
    
    def getOutline(self, isWrite : bool = True):
        for file in self.file_list:
            data = self.extractPDF(os.path.join(self.folder_path, file))
            if data == -1:
                print("File does not exists")
                continue
            data = pd.DataFrame(data)
            data['label'] = self.pipeline.predict(data)
            data.sort_values(by=['page', 'coord'], inplace=True)
            title, title_y1, title_list = self.getTitle(data)
            headings = data[data.label == 1]
            headings = headings[((headings.page != 1) | ((headings.page==1) & (headings.coord > title_y1)))]
            headings = self.levelHeadings(headings)
            # title = " ".join(x[1] for x in title_list)
            paragraphs = self.getParagraph(data, title_list, headings)
            
            json_data = {
                "title": title,
                "outline": [
                    {"level": f"H{row.heading_level}", "text": row.text, "page": row.page}
                    for row in headings.itertuples(index=False)
                ]
            }
            
            self.data[file] = data
            self.headings[file] = headings
            self.paragraphs.extend(paragraphs)
            
            if isWrite:
                output_file = os.path.join(self.output_path, f"{file[:-4]}.json")
                with open(output_file, "w") as f:
                    json.dump(json_data, f, indent=2)
            
        # e = pd.concat([val for key,val in self.data.items()], ignore_index=True)
        # return e
    
    def getAnalysis(self, input : str):
        self.file_list = []
        with open(os.path.join(self.folder_path, input)) as f:
            data = json.load(f)
        
        self.folder_path = os.path.join(self.folder_path, 'PDFs/')
        
        for i in data['documents']:
            self.file_list.append(i['filename'])
            
        persona = data['persona']['role']
        job = data['job_to_be_done']['task']
        
        output_file_name = data['challenge_info']['challenge_id']
        
        self.getOutline(False)
        
        time = datetime.now().isoformat()
        
        data = pd.concat([val for key, val in self.data.items()])
        headings = pd.concat([val for key, val in self.headings.items()])
        
        query = f"{persona} is asking, {job}"
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        extracted_sections = []
        if not headings.empty:
            heading_texts = [f"passage: {t}" for t in headings["text"]]
            heading_embeddings = self.model.encode(heading_texts, convert_to_tensor=True)
            heading_scores = util.cos_sim(query_embedding, heading_embeddings)[0]
            topk_headings_idx = torch.topk(heading_scores, k=min(5, len(headings))).indices.tolist()
            # print(topk_headings_idx)

            for rank, idx in enumerate(topk_headings_idx, 1):
                row = headings.iloc[idx]
                extracted_sections.append({
                    "document": row.file_name,
                    # "section_title": row["text"].encode('utf-8').decode('unicode_escape'),
                    "section_title": row["text"],
                    "importance_rank": int(rank),
                    "page_number": int(row["page"])
                })
                
        subsection_analysis = []
        if self.paragraphs:
            para_texts = [f"passage: {p}" for p in self.paragraphs]
            para_embeddings = self.model.encode(para_texts, convert_to_tensor=True)
            para_scores = util.cos_sim(query_embedding, para_embeddings)[0]
            topk_paras_idx = torch.topk(para_scores, k=min(5, len(para_texts))).indices.tolist()
            # print(topk_paras_idx)

            for idx in topk_paras_idx:
                subsection_analysis.append({
                    "document": self.paragraphs[idx][1],
                    # "refined_text": self.paragraphs[idx][2].encode('utf-8').decode('unicode_escape'),
                    "refined_text": self.paragraphs[idx][2],
                    "page_number": int(self.paragraphs[idx][0])
                })
        
        json_data = {
            "metadata" : {
                "input_documents" : self.file_list,
                "persona" : persona,
                "job_to_be_done": job,
                "processing_timestamp": time
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
        
        output_file = os.path.join(self.output_path, f"{output_file_name}.json")
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
