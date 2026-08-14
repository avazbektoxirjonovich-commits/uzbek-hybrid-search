"""Windows uchun WikiExtractor runner.

pip'dagi wikiextractor 3.0.6 ning process_dump() funksiyasi Windows'da
ishlamaydi: u shartsiz ravishda multiprocessing "fork" kontekstidan
foydalanadi (WikiExtractor.py:359), Windows'da esa faqat "spawn" bor.
"spawn"ga monkeypatch qilinsa ham, reduce_process ochiq fayl handle'ni
child processga pickle qilishga urinib yana qulaydi (TypeError: cannot
pickle '_io.TextIOWrapper').

Yechim: process_dump ni bitta-jarayonli (single-process) versiya bilan
almashtiramiz — xuddi shu Extractor klassidan foydalanadi, faqat
multiprocessing Queue/Process o'rniga oddiy for-loop ishlatadi.
"""
import sys

import wikiextractor.WikiExtractor as we

_SINGLE_PROCESS_DUMP = r'''
def process_dump(input_file, template_file, out_file, file_size, file_compress,
                 process_count, html_safe):
    global knownNamespaces
    global templateNamespace, templatePrefix
    global moduleNamespace, modulePrefix
    global urlbase

    urlbase = ''
    input = decode_open(input_file)

    for line in input:
        m = tagRE.search(line)
        if not m:
            continue
        tag = m.group(2)
        if tag == 'base':
            base = m.group(3)
            urlbase = base[:base.rfind("/")]
        elif tag == 'namespace':
            knownNamespaces.add(m.group(3))
            if re.search('key="10"', line):
                templateNamespace = m.group(3)
                templatePrefix = templateNamespace + ':'
            elif re.search('key="828"', line):
                moduleNamespace = m.group(3)
                modulePrefix = moduleNamespace + ':'
        elif tag == '/siteinfo':
            break

    if expand_templates:
        template_load_start = default_timer()
        if template_file and os.path.exists(template_file):
            logging.info("Preprocessing '%s' to collect template definitions: this may take some time.", template_file)
            file = decode_open(template_file)
            templates = load_templates(file)
            file.close()
        else:
            if input_file == '-':
                raise ValueError("to use templates with stdin dump, must supply explicit template-file")
            logging.info("Preprocessing '%s' to collect template definitions: this may take some time.", input_file)
            templates = load_templates(input, template_file)
            input.close()
            input = decode_open(input_file)
        template_load_elapsed = default_timer() - template_load_start
        logging.info("Loaded %d templates in %.1fs", templates, template_load_elapsed)

    if out_file == '-':
        output = sys.stdout
    else:
        nextFile = NextFile(out_file)
        output = OutputSplitter(nextFile, file_size, file_compress)

    logging.info("Starting SINGLE-PROCESS page extraction from %s.", input_file)
    extract_start = default_timer()

    page = []
    id = ''
    revid = ''
    last_id = ''
    ordinal = 0
    inText = False
    redirect = False
    title = ''
    for line in input:
        if '<' not in line:
            if inText:
                page.append(line)
            continue
        m = tagRE.search(line)
        if not m:
            continue
        tag = m.group(2)
        if tag == 'page':
            page = []
            redirect = False
        elif tag == 'id' and not id:
            id = m.group(3)
        elif tag == 'id' and id:
            revid = m.group(3)
        elif tag == 'title':
            title = m.group(3)
        elif tag == 'redirect':
            redirect = True
        elif tag == 'text':
            inText = True
            line = line[m.start(3):m.end(3)]
            page.append(line)
            if m.lastindex == 4:
                inText = False
        elif tag == '/text':
            if m.group(1):
                page.append(m.group(1))
            inText = False
        elif inText:
            page.append(line)
        elif tag == '/page':
            colon = title.find(':')
            if (colon < 0 or (title[:colon] in acceptedNamespaces) and id != last_id and
                    not redirect and not title.startswith(templateNamespace)):
                out = StringIO()
                Extractor(id, revid, urlbase, title, page).extract(out, html_safe)
                text = out.getvalue()
                out.close()
                output.write(text)
                last_id = id
                ordinal += 1
                if ordinal % 5000 == 0:
                    interval_rate = ordinal / (default_timer() - extract_start)
                    logging.info("Extracted %d articles (%.1f art/s)", ordinal, interval_rate)
            id = ''
            revid = ''
            page = []

    input.close()
    if output != sys.stdout:
        output.close()
    extract_duration = default_timer() - extract_start
    extract_rate = ordinal / extract_duration if extract_duration > 0 else 0
    logging.info("Finished SINGLE-PROCESS extraction of %d articles in %.1fs (%.1f art/s)",
                 ordinal, extract_duration, extract_rate)
'''

if __name__ == '__main__':
    # process_dump ni we modulining o'z globals()'i bilan qayta e'lon qilamiz,
    # shunda ichidagi bare-name (tagRE, Extractor, decode_open, ...) larning
    # hammasi moduldagi asl qiymatlarga to'g'ri bog'lanadi.
    exec(compile(_SINGLE_PROCESS_DUMP, 'process_dump_single', 'exec'), we.__dict__)

    sys.argv = [
        'WikiExtractor.py',
        'uzwiki-20251120-pages-articles-multistream.xml',
        '-o', 'wiki_output',
        '--json',
        '--processes', '1',
    ]
    we.main()
