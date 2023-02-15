#!/usr/bin/env python3
from sys import argv, stderr
from os import mkdir
from os.path import isdir, join, dirname
from nltk.corpus import BracketParseCorpusReader
from data.cross.dptb import read_tree, read_graph
from tqdm import tqdm
from data.cross.dag import XMLWriter

try:
    _, to_type, wsj_path, xml_fpath = argv
    to_type = to_type.lower()
    reader = BracketParseCorpusReader(wsj_path, r".*/wsj_.*\.mrg")
except:
    print('Invalid command: ', ' '.join(argv[1:]), file = stderr)
    print('Usage: to_gptb type path_to_WSJ FOLDER_or_XML_FILE_NAME', file = stderr)
    exit()

if to_type not in ('g', 'd', 'gptb', 'dptb'):
    print('Unsupported type:', to_type, file = stderr)
    print('  Supported type: d/dptb or g/gptb', file = stderr)
    exit()
else:
    convert = (read_tree, read_graph)[to_type[0] == 'g']

fileids = reader.fileids()
if one_xml := xml_fpath.endswith('.xml'):
    one_xml = XMLWriter()
else:
    if not isdir(xml_fpath):
        mkdir(xml_fpath)
    xml_files = {}
    xml_folders = set()
    for fileid in fileids:
        xml_files[fileid] = fpath = join(xml_fpath, fileid[:-4] + '.xml')
        if (fpath := dirname(fpath)) not in xml_folders:
            xml_folders.add(fpath)
            if not isdir(fpath):
                mkdir(fpath)

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
            except:
                continue

            (one_xml if one_xml else xml_writer).add(bt, td)
        if not one_xml:
            xml_writer.dump(xml_files[fileid])
        qbar.update(1)
if one_xml:
    one_xml.dump(xml_fpath)