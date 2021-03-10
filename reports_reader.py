
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

DATA_PATH = 'report_files'
START_YEAR = 2000
LAST_YEAR = 2020
YEARS = [str(i) for i in list(range(START_YEAR, LAST_YEAR))]

if __name__ == "__main__":

    company_names = os.listdir(DATA_PATH)