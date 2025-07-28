from PDFDataExtractor import PDFDataExtractor

extractor = PDFDataExtractor('./input', './models/RandomForest.joblib', './output')
extractor.getOutline()