import gspread
from pathlib import Path

filepath = Path('service_account.json')

gc = gspread.oauth(
    credentials_filename=filepath,
)
sh = gc.open("DCO AI Generated")
# worksheet.clear()
print(sh.sheet1.get('B1'))