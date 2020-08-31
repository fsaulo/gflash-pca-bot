count=0

ls -S --escape ../images/notes/* | while read f; do
    n=$(printf "%04d" $count)
    ((count++))
    mv --no-clobber "$f" ../images/classification/"$rename$n.png"
done
