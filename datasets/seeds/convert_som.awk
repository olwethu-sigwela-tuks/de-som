BEGIN{
	ORS = ""
	print "7"
}
{
	ORS = "
"
	print ""
	ORS = " "
	for (i = 1; i < NF; i++) {
		print 
	}
	if ( == "1") print "Kama"
	else if ( == "2") print "Rosa"
	else if ( == "3") print "Canadian"
}
