import json
import re


def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext


def match():
    with open("data.json", "r", encoding="utf-8") as fdata, \
            open("label.json", "r", encoding="utf-8") as flabel:
        data = json.load(fdata)
        label = json.load(flabel)
        labels = {}
        for l in label["label_2"]["data"]:
            labels[l["IdChuyenMuc"]]= l["TenChuyenMuc"]
        with open("examples.csv", "w", encoding="utf-8") as writer:
            for d in data["data"]:
                id_label = d['IdChuyenMuc']
                target = labels[id_label] if id_label in labels else id_label
                writer.write(f"{cleanhtml(d['TieuDe'])}\t{cleanhtml(d['NoiDung'])}\t{target}\n")
            writer.close()
        fdata.close()
        flabel.close()
        print(labels)


if __name__ == "__main__":
    match()