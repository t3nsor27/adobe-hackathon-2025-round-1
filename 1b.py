from PDFDataExtractor import PDFDataExtractor

extractor1 = PDFDataExtractor('./input/challenge_1b/Collection_1', './models/RandomForest.joblib', './output/challenge_1b/')
extractor1.getAnalysis('input.json')

extractor2 = PDFDataExtractor('./input/challenge_1b/Collection_2', './models/RandomForest.joblib', './output/challenge_1b/')
extractor2.getAnalysis('input.json')