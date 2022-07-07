PORT := 8080

run:
	streamlit run main.py --server.port $(PORT) --server.headless true
