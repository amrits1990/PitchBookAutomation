#!/usr/bin/env python3
"""
Create consolidated chunks JSON file from TranscriptRAG processing results
This script demonstrates how to extract all chunks with metadata into a single file
"""

import json
import os
from datetime import datetime
from get_transcript_chunks import get_transcript_chunks

def create_consolidated_chunks_file(tickers, start_date, years_back=2, output_file="consolidated_transcript_chunks.json"):
    """
    Process transcripts and create a consolidated chunks file
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date in YYYY-MM-DD format
        years_back: Number of years to look back
        output_file: Output filename for consolidated chunks
    """
    print(f"Processing {len(tickers)} tickers: {', '.join(tickers)}")
    print(f"Date range: {years_back} years back from {start_date}")
    
    # Get full transcript data
    results = get_transcript_chunks(
        tickers=tickers,
        start_date=start_date,
        years_back=years_back,
        return_full_data=True,
        chunk_size=800,
        overlap=150
    )
    
    if results['status'] != 'success':
        print(f"Error: {results.get('message', 'Unknown error')}")
        return None
    
    # Create consolidated chunks structure
    consolidated_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "source": "TranscriptRAG",
            "parameters": {
                "tickers": tickers,
                "start_date": start_date,
                "years_back": years_back,
                "chunk_size": 800,
                "overlap": 150
            },
            "summary": results['summary']
        },
        "transcripts": [],
        "all_chunks": []
    }
    
    total_chunks = 0
    global_chunk_counter = 0
    
    # Process each transcript
    for transcript_info in results['successful_transcripts']:
        transcript_summary = {
            "ticker": transcript_info['ticker'],
            "quarter": transcript_info['quarter'],
            "fiscal_year": transcript_info['fiscal_year'],
            "transcript_date": transcript_info['transcript_date'],
            "chunk_count": transcript_info['chunk_count'],
            "processed_at": transcript_info['processed_at']
        }
        
        consolidated_data['transcripts'].append(transcript_summary)
        
        # Add all chunks to consolidated list
        transcript_dataset = transcript_info['transcript_dataset']
        for chunk in transcript_dataset['chunks']:
            # Enhance chunk with transcript identification
            enhanced_chunk = {
                "global_chunk_id": global_chunk_counter,
                "transcript_ticker": transcript_info['ticker'],
                "transcript_quarter": transcript_info['quarter'],
                "transcript_fiscal_year": transcript_info['fiscal_year'],
                "transcript_date": transcript_info['transcript_date'],
                "chunk_id": chunk['chunk_id'],
                "text": chunk['text'],
                "length": chunk['length'],
                "metadata": chunk['metadata']
            }
            
            consolidated_data['all_chunks'].append(enhanced_chunk)
            global_chunk_counter += 1
            total_chunks += 1
    
    # Update metadata
    consolidated_data['metadata']['total_transcripts'] = len(results['successful_transcripts'])
    consolidated_data['metadata']['total_chunks'] = total_chunks
    
    # Save consolidated file
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(consolidated_data, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\n✓ Consolidated chunks saved to: {output_file}")
    print(f"✓ Total transcripts processed: {len(results['successful_transcripts'])}")
    print(f"✓ Total chunks created: {total_chunks}")
    
    # Print sample of chunks per transcript
    print(f"\nChunk distribution:")
    for transcript_info in results['successful_transcripts']:
        print(f"  - {transcript_info['ticker']} {transcript_info['quarter']} {transcript_info['fiscal_year']}: {transcript_info['chunk_count']} chunks")
    
    return output_file

def main():
    """Example usage of consolidated chunks creation"""
    
    # Example: Process Apple transcripts
    tickers = ['AAPL']
    start_date = '2024-01-01'
    years_back = 1
    
    output_file = './output/consolidated_transcript_chunks.json'
    
    print("=== TranscriptRAG Consolidated Chunks Creation ===")
    print()
    
    result_file = create_consolidated_chunks_file(
        tickers=tickers,
        start_date=start_date,
        years_back=years_back,
        output_file=output_file
    )
    
    if result_file:
        print(f"\n=== Processing Complete ===")
        print(f"Consolidated chunks file: {result_file}")
        print(f"This file contains all chunks with full metadata for RAG applications.")
        
        # Show sample chunk structure
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if data['all_chunks']:
            print(f"\nSample chunk structure:")
            sample_chunk = data['all_chunks'][0]
            print(f"  - Transcript: {sample_chunk['transcript_ticker']} {sample_chunk['transcript_quarter']} {sample_chunk['transcript_fiscal_year']}")
            print(f"  - Text length: {sample_chunk['length']} characters")
            print(f"  - Section: {sample_chunk['metadata'].get('section_name', 'Unknown')}")
            print(f"  - Speakers: {sample_chunk['metadata'].get('speakers', [])}")
            print(f"  - Text preview: {sample_chunk['text'][:100]}...")

if __name__ == "__main__":
    main()