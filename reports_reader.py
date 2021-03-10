
"""
The reported files format have the following rules:
1. We need a folder per company; this folder must include all the CSR reports of this company to consider
2. We have to include the reporting year in the file name using the format report_file_name_<year>.pfd
where year is a YYYY format, for instance: report_CSR_disney_2005.pfd, 2017disneycsrupdate.pdf,
report-2005-CSR-disney.pfd etc.
3. If you have the option, download Overviews reports instead of the entire reports, for instance:
https://us.pg.com/sustainability-reports/ offer both options.
"""

import os
import PyPDF2
import pandas as pd
DATA_PATH = 'report_files'
START_YEAR = 2000
LAST_YEAR = 2020
YEARS = [str(i) for i in list(range(START_YEAR, LAST_YEAR))]


def get_files_format(path):
    valid_formats = ['.pdf', '.txt']
    files = os.listdir(path)
    files = [file for file in files if os.path.isfile(os.path.join(path, file))]
    valid_files = []
    for valid_format in valid_formats:
        valid_files.extend([file for file in files if file.endswith(valid_format)])
    return valid_files


if __name__ == "__main__":

    company_names = os.listdir(DATA_PATH)
    company_names = [company_name for company_name in company_names
                     if os.path.isdir(os.path.join(DATA_PATH, company_name))]

    company_files = {company_name: get_files_format(path=os.path.join(DATA_PATH, company_name)) for company_name in company_names}
    df = pd.DataFrame(company_files)



