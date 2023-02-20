#!/usr/bin/env python3
from sys import argv, stderr
from os import mkdir
from os.path import isdir, join, dirname
from nltk.corpus import BracketParseCorpusReader
from data.cross import gap_degree, multi_attachment
from data.cross.dptb import read_tree, read_graph
from tqdm import tqdm
from data.cross.dag import XMLWriter
from collections import defaultdict

try:
    _, to_type, wsj_path, xml_fpath = argv
    to_type = to_type.lower()
    reader = BracketParseCorpusReader(wsj_path, r".*/wsj_.*\.mrg")
except:
    print('Invalid command: ', ' '.join(argv[1:]), file = stderr)
    print('Usage: ptb_to.py [d|g] [path_to_WSJ] [FOLDER_or_XML_FILE_NAME]', file = stderr)
    exit()

if to_type not in ('g', 'd', 'gptb', 'dptb'):
    print('Unsupported type:', to_type, file = stderr)
    print('  Supported type: d/dptb or g/gptb', file = stderr)
    exit()
else:
    if to_dag := (to_type[0] == 'g'):
        def graph_info(bt, td):
            return dict(max_gap = str(gap_degree(bt, td)), max_parent = str(max(multi_attachment(td).values())))
    else:
        def graph_info(bt, td):
            return dict(max_gap = str(gap_degree(bt, td)))
    convert = (read_tree, read_graph)[to_dag]

fileids = reader.fileids()
if one_xml := xml_fpath.endswith('.xml'):
    one_xml = XMLWriter()
    err_log = xml_fpath[:-4] + '.error'
else:
    if not isdir(xml_fpath):
        mkdir(xml_fpath)
    err_log = join(xml_fpath, '.error')
    xml_files = {}
    xml_folders = set()
    for fileid in fileids:
        xml_files[fileid] = fpath = join(xml_fpath, fileid[:-4] + '.xml')
        if (fpath := dirname(fpath)) not in xml_folders:
            xml_folders.add(fpath)
            if not isdir(fpath):
                mkdir(fpath)

count_suc = count_error = 0
error_log = defaultdict(list)
with tqdm(desc = wsj_path, total = len(fileids)) as qbar:
    for fileid in fileids:
        trees = reader.parsed_sents(fileid)
        total = len(trees)
        if not one_xml:
            xml_writer = XMLWriter()
        for eid, tree in enumerate(trees, 1):
            qbar.desc = f'Converting {fileid} ({eid}/{total})'
            qbar.update(0)
            try:
                bt, td = convert(tree)
                count_suc += 1
            except:
                error_log[fileid].append(str(eid))
                count_error += 1
                continue

            (one_xml if one_xml else xml_writer).add(bt, td, **graph_info(bt, td))
        if not one_xml:
            xml_writer.dump(xml_files[fileid])
        qbar.update(1)
    qbar.desc = f'Successful {count_suc} samples'
    if count_error:
        qbar.desc += f' and {count_error} errors (see {err_log})'
if one_xml:
    one_xml.dump(xml_fpath)

if error_log:
    with open(err_log, 'w') as fw:
        for fileid, eids in error_log.items():
            fw.write(fileid + '\t' + ','.join(eids) + '\n')