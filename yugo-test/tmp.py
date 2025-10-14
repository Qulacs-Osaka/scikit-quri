test = ["a", "b", "c", "d", "e"]
len_test = len(test)

for i in range(len_test):
    print("//ーーーーーーーーーーーーーーーーーーーーー")

    # 順方向で全要素出力
    for j in range(len_test):
        print(test[j])

    # 逆順で残り要素を出力
    for j in range(len_test - 1, i, -1):
        print(test[j])
