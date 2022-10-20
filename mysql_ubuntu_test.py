# -*- coding: utf-8 -*-


#from http.server import BaseHTTPRequestHandler, HTTPServer

import logging

import json

import mysql.connector

from mysql.connector import errorcode

import time
from datetime import datetime
  



def sendToDB():

    ##
    a = 124.0
    b = 22.2
    c = 32.0
    d = 42.0
    e = '200g_to_300g'
    f = 6.0
    ##
    try:

      #cnx = mysql.connector.connect(user='cihci', password='ab19696a',host='140.121.102.191',port='3306', database='ai_fish')#設定連線字串
      cnx = mysql.connector.connect(user='ai_account', password='R@BvwKXgN9*y)rsf',host='140.121.102.191',port='3306', database='ai_fish')
      mycursor = cnx.cursor() #開啟連線
      nowtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
      print("{}".format(nowtime))
      sql = "INSERT INTO fish_part (`time`, `length`, `height`, `weight`, `amount`, `class`, `total_weight`) VALUES(%s,%s,%s,%s,%s,%s,%s)"
      #t = str(datetime.fromtimestamp(message['timestamp']/1000))
      val = ( nowtime,a,b,c,d,e,f)

      mycursor.execute(sql, val)

      cnx.commit()

      print("pass 1 message , ID:", mycursor.lastrowid)

    except mysql.connector.Error as err:

      if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:

        print("Something is wrong with your user name or password")

      elif err.errno == errorcode.ER_BAD_DB_ERROR:

        print("Database does not exist")

      else:

        print("err is ",err)

    finally:

      cnx.close()
      print("saved success!")

def sendToDB_message(length,height,weight,amount,class_message,total_weight):

    ##
    a = 124.0
    b = 22.2
    c = 32.0
    d = 42.0
    e = '200g_to_300g'
    f = 6.0
    ##
    try:

      #cnx = mysql.connector.connect(user='cihci', password='ab19696a',host='140.121.102.191',port='3306', database='ai_fish')#設定連線字串
      cnx = mysql.connector.connect(user='ai_account', password='R@BvwKXgN9*y)rsf',host='140.121.102.191',port='3306', database='ai_fish')
      mycursor = cnx.cursor() #開啟連線
      nowtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
      print("{}".format(nowtime))
      sql = "INSERT INTO fish_part (`time`, `length`, `height`, `weight`, `amount`, `class`, `total_weight`) VALUES(%s,%s,%s,%s,%s,%s,%s)"
      #t = str(datetime.fromtimestamp(message['timestamp']/1000))
      # = ( nowtime,a,b,c,d,e,f)
      val = ( nowtime,length,height,weight,amount,class_message,total_weight)

      mycursor.execute(sql, val)

      cnx.commit()

      print("pass 1 message , ID:", mycursor.lastrowid)

    except mysql.connector.Error as err:

      if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:

        print("Something is wrong with your user name or password")

      elif err.errno == errorcode.ER_BAD_DB_ERROR:

        print("Database does not exist")

      else:

        print("err is ",err)

    finally:

      cnx.close()
      print("saved success!")


#message = [0, 1, 23.4, 12.4, 350, 14]
#sendToDB(message)
#sendToDB()


length = 24.0
height = 12.0
weight = 155.0
amount = 44.0
class_message = '200g_to_300g'
total_weight = 6.0


sendToDB_message(length,height,weight,amount,class_message,total_weight)
