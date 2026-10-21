BEGIN{
	ORS = ""
	print "6"
}
{
	ORS = "
"
	print ""
	ORS = " "
	for (i = 1; i <= NF; i++) {
		print 
	}
}
