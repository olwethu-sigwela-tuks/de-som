BEGIN{
	ORS = ""
	print "3"
	FS = ","
}
{
	ORS = "
"
	print ""
	ORS = " "
	for (i = 1; i < NF; i++) {
		print 
	}
	if ( == "1") print "survived"
	else if ( == "2") print "died"
}
