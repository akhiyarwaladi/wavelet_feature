# postgre_util.py

def insert(table,cols,vals,csr,debug=False):
  q = 'INSERT INTO '+table+' '+bracket(','.join(cols))
  q+= ' VALUES '+bracket(','.join([quote(i) for i in vals]))

  if debug: print q
  else: csr.execute(q)

def getID(col,table,colCond,cond,csr):
    q = 'SELECT '+col+' FROM '+table
    q+= ' WHERE '+colCond+'='+quote(cond)
    csr.execute(q);rows = csr.fetchall(); assert len(rows)==1
    ids = rows[0][0]
    return int(ids)

def quote(v):
  s = str(v)
  if not( isinstance(v, (int, long, float)) ):
    s = s.replace("'","''") # escape any single quote
    s = "'"+s+"'"
  return s

def bracket(s):
  return '('+s+')'
