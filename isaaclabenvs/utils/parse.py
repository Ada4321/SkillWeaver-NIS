def parse_indices(indices: str) -> list[int]:
    indices = "".join(char for char in indices if not char.isspace())
    result = []
    for period in indices.split(","):
        if "-" in period:
            start, end = period.split("-")
            result.extend(range(int(start), int(end)))
        else:
            result.append(int(period))
    return result

