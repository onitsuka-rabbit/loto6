# loto6
遺伝的アルゴリズムを使用したロト6予想システム

loto6.csv:過去の当選番号データ

setbias.csv:抽選に使用されるセット玉ごとに出やすい数字のデータ

LOTO6.py:遺伝的アルゴリズムを使用したロト6予想システム

このシステムでは、過去の当選番号、抽選に使われたセット玉から次回の当選番号を予想するというプログラムとなっています。
具体的には、あるセット玉において、過去に当選した番号が優先して選ばれるようになっています。この時、過去に1等、2等、3等、4等、5等として当選した番号の順に重みが重くなるようになっており、少なくとも5等が当たりやすくなるように予想を行います。
また、ここでの重みパラメータはプログラム内のreaN(Nは2～5の数字)変数を変更していただくだけで容易に変更可能となっています。
その他にも、ロト6のランダム性を考慮し、random.seed()を変更することによって少し予想結果に幅が出るようにしています。
拙いプログラムではございますが、もしよろしければご活用ください。

主な変更可能パラメータ:
予想するセット玉
today_setdata
当選等数優先度
rea2,rea3,rea4,rea5
遺伝回数
NGEN
突然変異確率
MUTINDPB


