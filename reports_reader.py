
"""

1. INPUT DATA STRUCTURE
We have to storage the pdf-reports according to the following rules :
0. The project must include a data directory called <report_files>
1. We need inside <report_files> a folder per company; this folder must include all the CSR reports to analyze
2. We have to include the reporting year in the file name using the format report_file_name_<year>.pdf
where year is a YYYY format, for instance: report_CSR_disney_2005.pfd, 2017disneycsrupdate.pdf,
report-2005-CSR-disney.pfd etc.
3. If you have the option, download Overviews reports instead of the entire reports, for instance:
https://us.pg.com/sustainability-reports/ offer both options.

This is an example of the report_files structure:
report_files
├── disney
    ├── 2017disneycsrupdate.pdf
    ├── 2018-CSR-Report.pdf
    ├── CSR2019Report.pdf
    ├── FY08Disney_CR_Report_2008.pdf
    ├── FY09Disney_FY09_CR_Update_2009.pdf
    └── FY10Disney_2010_CC_Report.pdf
├── microsoft
    ├── CitizenshipReport_Composite_LowRes_9-22-2005.pdf
    ├── Microsoft-2018-CSR-Annual-Report.pdf
    ├── Microsoft-2019-CSR-Annual-Report.pdf
    ├── Microsoft-2020-CSR-Annual-Report.pdf
    ├── citizenship2003.pdf
    ├──citizenship2004.pdf
└── p&g
    ├── 2000_Sustainability_Report_Overview.pdf
    ├── 2001_Sustainability_Report_Overview.pdf
    ├── 2002_Sustainability_Report_Overview.pdf
    ├── 2017_Citizenship_Report_Executive_Summary.pdf
    ├── 2018_P_G_Corporate_Citizenship_Report_Executive_Summary.pdf
    └── citizenship_report_2019_executive_summary.pdf


2. HOW WORKS
For this project, the PDF files that we will process include files from the year 2000 to 2020; it involves several
PDF formats (old and new style). Each document is processed through 3 pdf-scrapping packages to support the different
types of pdf formats (PyPDF4 -> PyPDF2 -> pdfminer). The reader script will apply all the libraries returning the
result with more text content.

You have only one parameter to set in this script, "page_indexes". It works as follow, imagine you have a pdf
document with 40 pages:
You can indicate positive indexes, for instance, [1,2,3] will return a corpus with the pages 1,2 and 3 of each report.
You can indicate negative indexex, for instance, [-1,-2,-3] will return a corpus with the last 3 pages 40, 39 and 38.
You can mix positive and negative indexes, for instance, [1,2,-1] will return a corpus with pages 1, 2 and 40.
If you want all the pages use page_indexes = [], but is not recommended, the main information is in the first and last
pages. All the pages could produce low quality topics in future steps.

By default, page_indexes = [0, 1, 2, -3, -2, -1], to process the first and last 3 pages of each report.

3. OUTPUT DATA STRUCTURE

At the end. the script will create a table where each row contains the information of a unique report, indicating
the company name, the file name, the reporting year, the pdf-report path and the extracted corpus for instance:

company_name:                                             microsoft
company_files:                 Microsoft-2019-CSR-Annual-Report.pdf
year:                                                          2019
file_path:        report_files/microsoft/Microsoft-2019-CSR-Annu...
corpus:           microsoft corporate social responsibility repo...
"""

import os
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from pdf_readers import pypdf4_reader, pypdf2_reader, pdf_miner_reader
from text_cleaner import text_cleaner

TEXT_FORMAT = '.txt'
PDF_FORMAT = '.pdf'
DATA_PATH = 'report_files'
START_YEAR = 1998
LAST_YEAR = 2021
YEARS = [str(i) for i in list(range(START_YEAR, LAST_YEAR+1))]
OUTPUT_PATH = 'text_corpus'


def get_files_format(path_directory):
    """
    Get from the path_directory all the valid files
    :param path_directory:
    :return:
    """
    valid_formats = [PDF_FORMAT, TEXT_FORMAT]
    files = os.listdir(path_directory)
    files = [file for file in files if os.path.isfile(os.path.join(path_directory, file))]
    valid_files = []
    for valid_format in valid_formats:
        valid_files.extend([file for file in files if file.endswith(valid_format)])
    return valid_files


def get_year_from_file_name(file_name):
    """
    Get the year of the report from the file_name
    :param file_name:
    :return:
    """
    year_ = None
    for year in YEARS:
        if year in file_name:
            year_ = year
            break
    return year_


class ReportsReader:
    """
    Get the text corpus of the file_path using several libraries considering the corpus with more text. This method
    will return the text of the pdf pages indicated on the attribute page_indexes.
    """
    def __init__(self, pdf_path, page_indexes):
        self.corpus = ''
        self.pdf_path = pdf_path
        self.page_indexes = page_indexes

    def pdf_to_text_pdf_miner(self):
        raw_corpus = pypdf2_reader(file_path=self.pdf_path, page_indexes=self.page_indexes)
        return text_cleaner(raw_corpus)

    def pdf_to_text_pypdf2(self):
        raw_corpus = pypdf4_reader(file_path=self.pdf_path, page_indexes=self.page_indexes)
        return text_cleaner(raw_corpus)

    def pdf_to_text_pypdf4(self):
        raw_corpus = pdf_miner_reader(file_path=self.pdf_path, page_indexes=self.page_indexes)
        return text_cleaner(raw_corpus)

    def get_max_corpus(self, corpus_method):
        corpus_size = len(self.corpus.split())
        corpus_candidate = corpus_method()
        corpus_candidate_size = len(corpus_candidate.split())
        print(self.pdf_path, corpus_method.__name__, corpus_candidate_size)
        return corpus_candidate if corpus_candidate_size > corpus_size else self.corpus

    def __call__(self):
        pdf_to_text_methods = [self.pdf_to_text_pdf_miner, self.pdf_to_text_pypdf2, self.pdf_to_text_pypdf4]
        if self.pdf_path.endswith(PDF_FORMAT):
            for pdf_to_text_method in pdf_to_text_methods:
                try:
                    self.corpus = self.get_max_corpus(pdf_to_text_method)
                except Exception as e:
                    print(f'Unexpected error for {self.pdf_path}:', e)
        return self.corpus


if __name__ == "__main__":

    start_time = datetime.now()
    company_names = os.listdir(DATA_PATH)
    company_names = [company_name for company_name in company_names
                     if os.path.isdir(os.path.join(DATA_PATH, company_name))]

    company_files = {company_name: get_files_format(path_directory=os.path.join(DATA_PATH, company_name)) for
                     company_name in company_names}

    company_data = {'company_name': [], 'company_files': []}

    for company_name, company_files in company_files.items():
        company_data['company_name'].extend([company_name]*len(company_files))
        company_data['company_files'].extend(company_files)

    df_reports_data = pd.DataFrame(company_data)
    df_reports_data['year'] = df_reports_data['company_files'].apply(lambda row: get_year_from_file_name(row))
    df_reports_data['file_path'] = df_reports_data.apply(lambda row: os.path.join(DATA_PATH,
                                                                                  row['company_name'],
                                                                                  row['company_files']), axis=1)
    page_indexes = [0, 1, 2, -3, -2, -1]  # Choose the page index you want
    tqdm.pandas()
    df_reports_data['corpus'] = df_reports_data['file_path'].apply(lambda pdf_path: ReportsReader(pdf_path,
                                                                                                  page_indexes)())
    file_name = 'csr_corpus_pages_index'+"_".join([str(page_index) for page_index in page_indexes])+'.csv'
    file_path = os.path.join(OUTPUT_PATH, file_name)
    df_reports_data.to_csv(file_path, sep=';', index=False)

    end_time = datetime.now()
    print(f'Duration: {end_time - start_time}')  # -> Duration: 3:50:50.830833

    # Files manual fixing
    """
    df_reports_data['year'] = df_reports_data['company_files'].apply(lambda row: get_year_from_file_name(row))
    df_reports_data['company_name'] = df_reports_data['company_name'].apply(lambda row: row.split('.')[0])
    """






