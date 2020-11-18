from os.path import join, isfile
from subprocess import Popen, PIPE
from multiprocessing import Pool
from tempfile import NamedTemporaryFile
try:
    from urllib import FancyURLopener
except ImportError:
    from urllib.request import FancyURLopener

# list of currently supported representations
# (note that in the online CoreNLP demo, 'collapsed' is called 'enhanced')
REPRESENTATIONS = ('basic', 'collapsed', 'CCprocessed', 'collapsedTree')
DEFAULT_CORENLP_VERSION = '4.2.0'
JAVA_CLASS_NAME = 'edu.stanford.nlp.trees.EnglishGrammaticalStructure'

class JavaRuntimeVersionError(EnvironmentError):
    """Error for when the Java runtime environment is too old to support
    the specified version of Stanford CoreNLP."""
    def __init__(self):
        message = "Your Java runtime is too old (must be 1.8+ to use " \
                  "CoreNLP version 3.5.0 or later and 1.6+ to use CoreNLP " \
                  "version 1.3.1 or later)"
        super(JavaRuntimeVersionError, self).__init__(message)

class ErrorAwareURLOpener(FancyURLopener):
    def http_error_default(self, url, fp, errcode, errmsg, headers):
        raise ValueError("Error downloading %r: %s %s" %
                         (url, errcode, errmsg))

def _unit(x):
    hyp = x.rfind('-')
    return x[:hyp], int(x[hyp + 1:])

class StanfordDependencies:
    def __init__(self,
                 install_dir,
                 jar_filename        = None,
                 download_if_missing = True,
                 version             = DEFAULT_CORENLP_VERSION,
                 java_command        = 'java',
                 print_func          = print):
        if not (jar_filename or version is not None or download_if_missing):
            raise ValueError("Must set either jar_filename, version, "
                             "or download_if_missing to True.")

        if jar_filename is None:
            jar_filename = join(install_dir, f'stanford-corenlp-{version}.jar')

        if not isfile(jar_filename) and download_if_missing:
            jar_url = self.get_jar_url(version)
            print_func("Downloading %r -> %r" % (jar_url, jar_filename))
            opener = ErrorAwareURLOpener()
            opener.retrieve(jar_url, filename = jar_filename)

        assert isfile(jar_filename)
        self.jar_filename = jar_filename
        self.java_command = java_command

    def convert_trees(self,
                      str_trees,
                      representation = 'basic',
                      include_punct  = True,
                      include_erased = False,
                      universal      = True):
        assert representation in REPRESENTATIONS, 'Unknown representation: {representation}'
        with NamedTemporaryFile(delete = False) as input_file:
            command = [self.java_command,
                       '-ea',
                       '-cp', self.jar_filename,
                       JAVA_CLASS_NAME,
                       '-' + representation,
                       '-treeFile', input_file.name]
            if include_punct or include_erased:
                command.append('-keepPunct')
            if not universal:
                command.append('-originalDependencies')

            input_file.write(str_trees.encode('utf-8'))
            input_file.flush()
            input_file.close()

            # if we're including erased, we want to include punctuation
            # since otherwise we won't know what SD considers punctuation
            sd_process = Popen(command, stdout = PIPE, stderr = PIPE, universal_newlines = True)
            return_code = sd_process.wait()
        stderr = sd_process.stderr.read()
        stdout = sd_process.stdout.read()
        StanfordDependencies._raise_on_bad_exit_or_output(return_code, stderr)

        # picks out (deprel, gov, govindex, dep, depindex) from Stanford
        # Dependencies text (e.g., "nsubj(word-1, otherword-2)")

        dep_head = None
        batch = []
        for line in stdout.splitlines():
            if line:
                lhs = line.find('(')
                rhs = line.rfind(')')
                head, dep = line[lhs + 1: rhs].split(', ')
                head, hid = _unit(head)
                dep, dpid = _unit(dep)
                # try:
                # except:
                #     import pdb; pdb.set_trace()
                # try:
                # except:
                #     import pdb; pdb.set_trace()
                # (deprel, gov_form, head, gov_is_copy, form, index,
                # dep_is_copy) = matches[0]
                if dep_head is None:
                    dep_head = {dpid: hid}
                else:
                    dep_head[dpid] = hid
            else:
                batch.append(dep_head)
                dep_head = None
        if dep_head:
            batch.append(dep_head)
        return batch

    def convert_corpus(self, trees, n = 50):
        batch = None
        batches = []
        for x in trees:
            if batch is None:
                batch = []
            batch.append(x)
            if len(batch) == n:
                batches.append('\n'.join(' '.join(str(tree).split()) for tree in batch))
                batch = None
        if batch:
            batches.append('\n'.join(' '.join(str(tree).split()) for tree in batch))
        corpus = []
        with Pool() as p:
            for batch in p.map(self.convert_trees, batches):
                corpus.extend(batch)
        if False:
            print('final check')
            assert len(corpus) == len(trees)
            for dep_head, tree in zip(corpus, trees):
                words = [w for w, t in tree.pos() if t != '-NONE-']
                assert len(words) == len(dep_head)
                # for dep, head in sorted(dep_head.items(), key = lambda x: x[0]):
                #     print(words[dep - 1].rjust(38), ' -> ', words[head - 1] if head > 0 else 'ROOT')
                # import pdb; pdb.set_trace()
        return corpus

    @staticmethod
    def _raise_on_bad_exit_or_output(return_code, stderr):
        if 'PennTreeReader: warning:' in stderr:
            raise ValueError("Tree(s) not in valid Penn Treebank format")

        if return_code:
            if 'Unsupported major.minor version' in stderr:
                # Oracle Java error message
                raise JavaRuntimeVersionError()
            elif 'JVMCFRE003 bad major version' in stderr:
                # IBM Java error message
                raise JavaRuntimeVersionError()
            else:
                raise ValueError('Bad exit code from Stanford CoreNLP')

    @staticmethod
    def get_jar_url(version = None):
        """Get the URL to a Stanford CoreNLP jar file with a specific
        version. These jars come from Maven since the Maven version is
        smaller than the full CoreNLP distributions. Defaults to
        DEFAULT_CORENLP_VERSION."""
        if version is None:
            version = DEFAULT_CORENLP_VERSION

        if not isinstance(version, str):
            raise TypeError(f'Version must be a string or None (got {version}).')
        jar_filename = f'stanford-corenlp-{version}.jar'
        return 'http://search.maven.org/remotecontent?filepath=' + \
               f'edu/stanford/nlp/stanford-corenlp/{version}/{jar_filename}'