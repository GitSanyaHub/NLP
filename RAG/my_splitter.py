import re


class MyRecursiveTextSplitter():
    def __init__(self, chunk_size=200, chunk_overlap=100, separators=None):
        self.separators = separators or ["\n\n", "\n", " ", ""]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def split_text(self, text, separator=None):
        separators = self.separators
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        for i, s in enumerate(separators):
            sepr = re.escape(s)
            if s == "":
                separator = s
                break
            if re.search(sepr, text):
                separator = s
                new_separators = separators[i + 1:]
                break

        sepr = re.escape(separator)
        splits = self.split_text_with_regex(text, sepr)

        good_splits = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = self.merge_splits(good_splits, separator)
                    final_chunks.extend(merged_text)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_splits = self.split_text(s, new_separators)
                    final_chunks.extend(other_splits)
        if good_splits:
            merged_text = self.merge_splits(good_splits, separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def merge_splits(self, splits, separator):
        all_chunks = []
        curr_chunk = []
        total = 0
        for s in splits:
            len_split = len(s)
            if total + len_split > self.chunk_size:
                if len(curr_chunk) > 0:
                    doc = separator.join(curr_chunk)
                    all_chunks.append(doc)
                    while total > self.chunk_overlap:
                        total -= len(curr_chunk[0]) + len(separator)
                        curr_chunk = curr_chunk[1:]
            curr_chunk.append(s)
            total += len_split + len(separator)
        if curr_chunk:
            doc = separator.join(curr_chunk)
            all_chunks.append(doc)
        return all_chunks

    def split_text_with_regex(self, text, separator):
        if separator:
            splits = re.split(separator, text)
        else:
            splits = list(text)
        return [s for s in splits if s]


