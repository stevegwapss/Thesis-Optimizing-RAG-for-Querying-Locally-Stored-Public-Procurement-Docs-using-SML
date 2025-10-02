# download_model.py - Pre-download models for offline LangChain usage

print('Downloading HuggingFace embeddings model for LangChain...')

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    
    # This will download and cache the model locally
    print('Initializing HuggingFace embeddings...')
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/paraphrase-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    
    print(' Model downloaded and cached successfully!')
    print(' Model location: ~/.cache/huggingface/')
    print(' The model is now available for offline use in the RAG system')
    
    # Test the embeddings to ensure they work
    test_text = 'This is a test document for procurement system.'
    test_embedding = embeddings.embed_query(test_text)
    print(f' Model test successful - Generated embedding vector of size: {len(test_embedding)}')
    
except ImportError as e:
    print(f' Import error: {e}')
    print(' Make sure you have installed: pip install langchain-huggingface sentence-transformers')
    
except Exception as e:
    print(f' Error downloading model: {e}')
    print(' Check your internet connection and try again')

print('\n You can now run your RAG system offline!')
print(' Next: python app.py')
