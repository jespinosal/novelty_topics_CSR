

LDA_PATHS = {'lda_model': 'results/lda_model',
             'lda_dict': 'results/lda_dictionary',
             'text_corpus': 'text_corpus/csr_corpus_pages_index0_1_2_-3_-2_-1.csv',
             'reports': 'report_files',
             'topics_distribution': 'results/ESG_topics_distribution.csv',
             'lda_model_performance': 'results/lda_model_performance.csv',
             'company_topics_efforts': 'results/df_company_topics.csv'}

LDA_CONFIG = {'topics_range': (90, 110),
              'chunksize': 100,
              'pases': 10}

TOPICS_KEY_WORDS = {'environmental': "sustainability energy water decarbonization green weather climate crisis extreme "
                                     "wildfire floods raw materials sourcing production minerals footprint reduce waste"
                                     " smart efficient carbon emissions conserve conservation foster packaging "
                                     "renewable electricity resource restore planet forestry consumption innovations "
                                     "plastic recycle reusable petroleum aluminum refillable bottle manufacturing "
                                     "neutral efficiency circular tree wood deforestation healthy nature wildlife "
                                     "greenhouse storage recovery rivers ocean intelligent digital technology use "
                                     "temperature oil",
                    'social': "human rights people community diversity inclusion affirmative action workforce "
                              "demographics donation volunteer pro bono equality honor individuality employee self "
                              "safe voice growth equal culture social norms stereotypes bias conversations motivate "
                              "LGBTQ disabilities gender ethic racial health",
                    'corporate governance': "corporate governance shareholders accountability transparency " 
                                            "decision-making board of directors committees owners appropriately " 
                                            "structure objective monitor safeguards stewardship authority practices" 
                                            " operations independent align interests comply requirements law "
                                            "composition selection compensation succession leadership evaluation risk "
                                            "oversight director audit charter financial expert nominating regulatory "
                                            "public policy elections tenure vote shares equity grant approval "
                                            "compliance standards conduct duties chair principles review assessment "
                                            "politics statement rules mission "}
