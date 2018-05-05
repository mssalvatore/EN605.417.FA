<?php

for ($x = 0; $x < 5; $x++){
$limit = rand(3,10);
for ($i = 0; $i < $limit; $i++) {
    $limit1 = rand(5,12);
    for ($j = 0; $j < $limit1; $j++) {
        print rand(1,50) . " ";
    }
    print "\n";
}
    print "\n";
}
