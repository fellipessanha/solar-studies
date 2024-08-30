import os
import wget
from threading import Thread


def downloadDataFromYear(year):
    filename = f"{year}.zip"
    if filename in os.listdir("inmet-data"):
        print(f"{filename} already downloaded!")
        return

    wget.download(
        f"https://portal.inmet.gov.br/uploads/dadoshistoricos/{filename}",
        f"inmet-data/{filename}",
    )


threads = []
for year in range(2000, 2025):
    t = Thread(target=downloadDataFromYear, args=(year,))
    threads.append(t)
    t.start()

for t in threads:
    print(f"joining thread {t}")
    t.join()
