# novelty_topics_CSR
This project implements a measure to detect and qualify the novelty of the firm's CSR (corporate social responsibility)
strategy.  The project is based on the U.S.A major corporations, processing the CSR reports available on the 
companie's websites. 


Instructions:

1. Set up project (choose 1.1 or 1.2:
1.1 From Pycharm: 
    File->New Project (choose the folder "novelty_topics_CSR)
1.2 From shell: 
    export PYTHONPATH="/Users/jonathanespinosa/Projects/portfolio/novelty_topics_CSR"
    echo $PYTHONPATH

2. Install requirements:
2.1 Install automatic libraries from shell: 
    pip install requirements.txt
2.2 Install manual requirements: 
    2.1 Install nltk stopwords from python session: 
    >> import nltk
    >> nltk.download('stopwords') 

    2.2 Download Spacy models, run from shell:
    python -m spacy download en_core_web_sm
    
3. Generate the text corpus, execute the script: reports_reader.py
This script will generate --> text_corpus/csr_corpus_pages_index0_1_2_-3_-2_-1.csv

4. Generate lda model, execute the script: topic_model/lda_model.py
This script will generate the lda_model files:
results/lda_dictionary
results/lda_model
results/lda_model.expElogbeta.npy
results/lda_model.id2word
results/lda_model.state

And also will generate some report .csv files:
results/ESG_topics_distribution.csv
results/lda_model_performance.csv
results/top_words.csv

5. Generate the LDA model analysis, to perform predictions and analysis.
Execute: topic_model/topics_analysis.py
This script will produce: results/df_company_topics.csv


