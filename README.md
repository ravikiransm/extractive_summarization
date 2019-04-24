# extractive_summarization
Extractive summarization using sentence scoring

### requirements
    python == 3.6
    spacy == 2.0.13
    nltk >= 3.0
    
Set the DIR_PATH in config.py     

### Run
     run.py

### Sentence Scoring Method

After preprocessing the input document is segmented into
collection of words in which each word has its individual
frequency.
The sentences are ranked based on important features:


    	1. Frequency
            2. Sentence Position
            3. Cue words
            4. Sentence length.
            
After each sentence is scored they are arranged in
descending order of their score value i.e. the sentence
whose score value is highest is in top position and the
sentence whose score value is lowest is in bottom position.

After ranking the sentences based on their total score the
summary is produced selecting certain number of top
ranked sentences where the number of sentences required is
provided by the user. For the reader’s convenience, the
selected sentences in the summary are reordered according
to their original positions in the document.
            
            
