/*
Copyright (C) 2011 by the Computer Poker Research Group, University of Alberta
*/

#ifndef _NET_H
#define _NET_H

#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>


#define READBUF_LEN 4096
#define NUM_PORT_CREATION_ATTEMPTS 10


/* buffered I/O on file descriptors

   Yes... this is basically re-implementing bits of a standard FILE.
   Unfortunately, trying to mix timeouts and FILE streams either
   a) doesn't work, or b) is fairly system specific */
typedef struct {
  int fd;
  int bufStart;
  int bufEnd;
  char buf[ READBUF_LEN ];
} ReadBuf;


/* open a socket to hostname/port
   returns file descriptor on success, <0 on failure */
int connectTo( char *hostname, uint16_t port );

/* try opening a socket suitable for connecting to
   if *desiredPort>0, uses specified port, otherwise use a random port
   returns actual port in *desiredPort
   returns file descriptor for socket, or -1 on failure */
int getListenSocket( uint16_t *desiredPort );


/* create a read buffer structure
   returns 0 on failure */
ReadBuf *createReadBuf( int fd );

/* destroy a read buffer - like fdopen, it will close the file descriptor */
void destroyReadBuf( ReadBuf *readBuf );

/* get a newline terminated line and place it as a string in 'line'
   terminates the string with a 0 character
   if timeoutMicros is non-negative, do not spend more than
   that number of microseconds waiting to read data
   return number of characters read (including newline, excluding 0)
   0 on end of file, or -1 on error or timeout */
ssize_t getLine( ReadBuf *readBuf,
		 size_t maxLen,
		 char *line,
		 int64_t timeoutMicros );


#endif
