import mwxml
import mwparserfromhell

dump_file = "arwiki-latest-pages-articles.xml"
output_file = "arwiki_plaintext.txt"

with open(output_file, 'w', encoding='utf-8') as out:
    dump = mwxml.Dump.from_file(open(dump_file, 'rb'))
    for page in dump:
        if page.namespace != 0:  # فقط المقالات، بدون صفحات الميتا مثل User/Template
            continue
        for revision in page:
            wikicode = mwparserfromhell.parse(revision.text or "")
            text = wikicode.strip_code()  # يحذف التنسيقات ويترك النص فقط
            if text.strip():
                out.write(f"\n### {page.title} ###\n")
                out.write(text + "\n")
            break  # نأخذ فقط أول نسخة من كل مقال

