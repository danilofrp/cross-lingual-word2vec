import json


class Translator():
    
    def __init__(self, en_pt_dict_path, pt_en_dict_path):
        self.dicts = dict()

        print("[ ] Loading EN-PT dict")
        with open(en_pt_dict_path, "rb") as f:
            self.dicts["en-pt"] = json.loads(f.read().decode("utf-8"))

        print("[ ] Loading PT-EN dict")
        with open(pt_en_dict_path, "rb") as f:
            self.dicts["pt-en"] = json.loads(f.read().decode("utf-8"))


    def translate(self, word, src_lang, dest_lang):
        try:
            return self.dicts[f"{src_lang}-{dest_lang}"][word].lower()
        except:
            return word


    def bulk_translate(self, words, src_lang, dest_lang):
        return [self.translate(word, src_lang, dest_lang) for word in words]


if __name__ == "__main__":
    
    from pathlib import Path

    base_path = Path(".")/"data"/"dicts"
    en_pt_dict_path = base_path / "en-pt.json.sample"
    pt_en_dict_path = base_path / "pt-en.json.sample"

    translator = Translator(en_pt_dict_path, pt_en_dict_path)
    print(translator.translate("hello", "en", "pt"))
    print(translator.translate("world", "en", "pt"))