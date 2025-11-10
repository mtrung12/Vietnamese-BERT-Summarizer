from vncorenlp import VnCoreNLP
import config

print("Initializing VnCoreNLP") 

mvncorenlp = VnCoreNLP(
    config.VNCORENLP_PATH,
    annotators="wseg",
    max_heap_size='-Xmx2g'
)

# You can even define the function that USES it right here
def mvn_word_tokenize(text: str) -> str:
    """Tokenizes text using the global VnCoreNLP instance."""
    tokenized = mvncorenlp.tokenize(text)
    flat = [token for sent in tokenized for token in sent]
    return " ".join(flat)