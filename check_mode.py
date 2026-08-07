import requests
from knowledge.store import find_chunks_by_source, get_collection

resp = requests.post("http://127.0.0.1:8000/knowledge/ingest", json={
    "text": "Throwaway text for endpoint doc_type verification, testing the JSON ingest route.",
    "source": "test_endpoint_doc_type",
    "doc_type": "resume",
})
print(resp.status_code, resp.json())

chunks = find_chunks_by_source("test_endpoint_doc_type")
for c in chunks:
    print(c["metadata"].get("doc_type"))

# cleanup
ids = [c["id"] for c in chunks]
if ids:
    get_collection().delete(ids=ids)
print("cleaned up")