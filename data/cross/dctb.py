from data.cross.dptb import _read_graph, rename_node_fn, shift_node_fn #, direct_read
# from data.cross import draw_str_lines

change_trace = {
    "但是 *pro* 把 我 妈 自己 留 *-1 家里 看 孩子 ， 你 觉得 *T*-1 好 吗": (((1,1,1,1,0,1,0,0), '-2'), ((1,1,1,0), '-2')),
    "「 *OP* 街坊 邻居 现在 最 常 一起 做 *T*-1 的 娱乐 就 是 到 俱乐部 来 运动 。 *PRO* 维持 身材 苗条 除了 *pro* 比较 不 会 有 *OP* *T*-2 肥胖 伴生 的 疾病 外 ， 也 会 让 自己 *PRO* 看 起来 更 年轻 ！ 」 王 太太 说 *T*-2 。": ((0, '-3'), ((4,1,0,0), '-3')),
    "“ 不仅 如此 ， 每 当 政治 运动 来临 ， *OP* *T*-1 象 我们 这样 出身 不 好 的 人 不管 平时 表现 多 好 ， 总 免不了 受到 冲击 ， *pro* *pro* 轻 则 停职 反省 ， *pro* *pro* 重 则 受 审查 、 挨 批判 ， ” *OP* *T*-2 现 已 七十三 岁 的 北京 中国 人民 大学 人口 研究所 教授 查瑞传 回忆 说 *-3 。": (((0, '-3')), ((2,1,1,0,0), 'T*-3')),
    "*pro* 致使 本人 8.2 亩 土地 使用权 *PRO* 被 霸占 *-1 ， *OP* 市政府 给予 *T*-2 的 每 年 每 亩 1300 元 的 土地 租金 被 *OP* 村委会 截留 *T*-1 ， *OP* *pro* 经营 *T*-4 *pro* 长 达 23 年 之 久 、 *T*-4 投资 巨大 现 市值 百万 的 农场 &lt; *pro* 有 录像 和 照片 》 ， 被 逼拆 *-3 而 不 予 *-3 补偿 ， 本人 不 同意 补偿 方案 就 遭受 殴打 的 不幸 遭遇 《 110 有 记录 》 。": (((0,1,1), '-5'), ((0,1,2,1,1,1,0,0), '-5')),
    "*pro* 理解 错 了 ， *pro* 需要 ， *pro* 把 行程单 和 邀请函 打印 出来 *RNR*-1 带 着 *-1 。": (((4,1,1,0), '-1'),)
}


def fix_for_ctb(tree):
    sent = ' '.join(tree.leaves()).strip()
    if sent in change_trace:
        for path, tid in change_trace[sent]:
            if isinstance(sub_tree := tree[path], str):
                # print(sub_tree)
                tr, _ = sub_tree.split('-', 1)
                tree[path] = tr + tid
            else:
                label = sub_tree.label()
                # print(label)
                if label[-1].isdigit():
                    label, _ = label.rsplit('-', 1)
                rename_node_fn(tree, path, label + tid)
        if sent.startswith('*pro* 致使 本人'):
            shift_node_fn(tree, (4, 0, 3), (4, 1))
            shift_node_fn(tree, (4, 4), (5,))
            shift_node_fn(tree, (4, 3), (5,))
            shift_node_fn(tree, (4, 2), (5,))
            # shift_node_fn(tree, (4, 0, 2), (5,))

        # print('\n'.join(draw_str_lines(*direct_read(tree))))

def read_graph(tree, return_type_count = False):
    fix_for_ctb(tree)
    return _read_graph(tree, True, return_type_count)