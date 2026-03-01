def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    # Split the array into two halves
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    # Merge the sorted halves
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    # Merge while both arrays have elements
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Add remaining elements from left or right
    result.extend(left[i:])
    result.extend(right[j:])

    return result

# Test with the given array
test_array = [38, 27, 43, 3, 9, 82, 10]
sorted_array = merge_sort(test_array)
print(sorted_array)
