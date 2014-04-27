# -*- encoding: utf-8
import sys
import StringIO
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re
import datetime
import codecs

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams
from pdfminer.converter import XMLConverter

class LineSegment(object):

    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def intersects(self, line):
        """Returns True if the given line segment intersects this one, False otherwise."""
            
        p = [self.x1 - 0, self.y1 - 0]
        q = [line.x1 - 0, line.y1 - 0]
        r = [self.x2 - self.x1, self.y2 - self.y1]
        s = [line.x2 - line.x1, line.y2 - line.y1]

        #segments are parallel
        if np.cross(r, s) == 0 :
            #segments are colinear
            if np.cross(np.subtract(q, p), r) == 0:
                return True
            #segments never intersect
            return False

        t = np.cross(np.subtract(q, p), s) / float(np.cross(r, s))
        u = np.cross(np.subtract(q, p), r) / float(np.cross(r, s))

        #segments intersect
        if (0 <= t <= 1) and (0 <= u <= 1):
            return True

        #segments never intersect
        return False

class Bbox(object):
    """Co-ordinates in this bbox are transformed from an origin of bottom-left
       to an origin at top-right."""

    def __init__(self, node, page_height=None, scale=10):
        bbox_str = node.get('bbox')
        bbox = [int(float(i) * scale) for i in bbox_str.split(',')]
        self._x1, self._y1, self._x2, self._y2 = bbox
        self.height = self._y2 - self._y1
        self.width = self._x2 - self._x1
        self.page_height = self.height if not page_height else page_height

    @property
    def x1(self): return self._x1

    @property
    def y1(self): return self.page_height - self._y2

    @property
    def x2(self): return self._x2

    @property
    def y2(self): return self.page_height - self._y1

class TextLine(Bbox):

    def __init__(self, textline, **kwargs):
        super(TextLine, self).__init__(textline, **kwargs)
        self.textline = textline

    def __str__(self):
        return '[{%s,%s}, {%s, %s}]' % tuple([self.x1, self.y1, self.x2, self.y2])

    @property
    def text(self):
        return ''.join([t.text for t in self.textline.iter('text')]).strip()

    @property
    def left_ray(self):
        return LineSegment(self.x1, self.y1 + self.height / 2, 0, self.y1 + self.height / 2)

    @property
    def right_ray(self):
        return LineSegment(self.x2, self.y1 + self.height / 2, self.page_height, self.y1 + self.height / 2)

    @property
    def top_border(self):
        return LineSegment(self.x1, self.y1, self.x2, self.y1)

    @property
    def bottom_border(self):
        return LineSegment(self.x1, self.y2, self.x2, self.y2)

    @property
    def left_border(self):
        return LineSegment(self.x1, self.y1, self.x1, self.y2)

    @property
    def right_border(self):
        return LineSegment(self.x2, self.y1, self.x2, self.y2)

class Page(Bbox):

    def __init__(self, page, **kwargs):
        super(Page, self).__init__(page, **kwargs)
        self.page = page                
        self._create_nodes()
        self._create_lines()

    def _create_nodes(self):
        self.nodes = []
        for textline in self.page.iter('textline'):
            self.nodes.append(TextLine(textline, page_height=self.height))
        self._get_leftmost_nodes()      

    def _get_leftmost_nodes(self):
        self.leftmost_nodes = []
        for node in self.nodes:
            #if line from middle left border hits any other nodes...then this is not a leftmost node.            
            for other in self.nodes:               
                if other == node:                    
                    continue
                if node.left_ray.intersects(other.left_border) or node.left_ray.intersects(other.right_border):                    
                    break                    
            else:
                self.leftmost_nodes.append(node)

    def _create_lines(self):
        """Create groups of all the boxes in the same line, ordered by y-position on page."""
        self.lines = []
        for node in self.leftmost_nodes:
            line = [node]
            for other in self.nodes:
                if other == node:
                    continue
                if node.right_ray.intersects(other.left_border) and node.right_ray.intersects(other.right_border):
                    line.append(other)
            line = sorted(line, key=lambda x : x.x1)
            self.lines.append(line)

        self.lines = sorted(self.lines, key=lambda x : min([n.y1 for n in x]))

class AIBTransaction(object):

    def __init__(self, date=None, details=None, amount=None, transaction_type=None, balance=None):
        self.date = date.strip()
        self.details = details.strip()
        self.amount = amount
        self.transaction_type = transaction_type
        self.balance = balance

    def __str__(self):
        display = '%-20s%-80s%15s%15.2f%15s' % (self.date, self.details, self.transaction_type, self.amount, self.balance)
        return display

class AIBTransactionTable(Page):
    
    def __init__(self, page, **kwargs):
        super(AIBTransactionTable, self).__init__(page, **kwargs)
        table_header_index, table_footer_index = self._find_table_boundaries()
        self.rows = self.lines[table_header_index:table_footer_index]
        self.headers = {
            'debit' : self.rows[0][2],
            'credit' : self.rows[0][3],
            'balance' : self.rows[0][4] 
        }
        self.parse_table()
        
    def __str__(self):
        return '\n'.join([str(t) for t in self.transactions])
        
    def _find_table_boundaries(self):
        start_of_table = ur'Date\s+Details\s+Debit €\s+Credit €\s+Balance €'
        end_of_table = ur'For Important Information and Standard Conditions'

        table_header_index = table_footer_index = None
        for i, line in enumerate(self.lines):
            text = '\t'.join([box.text for box in line])            
            if re.match(start_of_table, text):
                table_header_index = i
            if re.match(end_of_table, text):
                table_footer_index = i

        if table_header_index is None or table_footer_index is None:
            raise Exception('Couldn\'t find table boundary.')

        return table_header_index, table_footer_index

    def _get_box_type(self, box):
        for k,v in self.headers.items():
            if (v.x1 <= box.x1 <= v.x2) or (v.x1 <= box.x2 <= v.x2):
                return k
        return None

    def _parse_date(self, date_details):
        result = re.findall(r'^(\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4})(.*)', date_details)
        if not result:
            return None, date_details
        return result[0]

    def _parse_row(self, columns, num_rows, row_index):
        
        amount = box_type = balance = None
        #For each column in the row, after the details
        for i, box in enumerate(columns):            
            if re.match(r'\d+\.\d{2}', box.text):
                #We only care about the first one (debit or credit)
                if i == 0:
                    box_type = self._get_box_type(box)
                    amount = float(box.text)
                #Otherwise we're probably looking at a balance figure    
                else:
                    balance_check = self._get_box_type(box)
                    balance = float(box.text)
                #Balance-only rows should only occur at the start of the table. Otherwise they belong to the previous transaction.
                if box_type == 'balance' and row_index != 0:
                    balance = amount
                    return None, box_type, balance               
                if i != 0 and not balance_check:
                    raise Exception('box has no header, despite being numerical')
            #If our column is non-numerical, then we're done with this row.
            else:
                return amount, box_type, balance
        
        return amount, box_type, balance

    def parse_table(self):

        self.transactions = []
        
        financial_rows = self.rows[1:]
        for i, row in enumerate(financial_rows):
            date_details = row[0].text

            if row[0].x1 > self.headers['debit'].x1:
                continue

            if len(row) == 1:
                self.transactions[-1].details = ' '.join([self.transactions[-1].details, date_details])
                continue

            financial_columns = row[1:]            
            amount, box_type, balance = self._parse_row(financial_columns, len(financial_rows), i)

            if not amount and not box_type and not balance:
                self.transactions[-1].details = ' '.join([self.transactions[-1].details, date_details])
                continue
            elif not amount and box_type == 'balance' and self.transactions:
                self.transactions[-1].details = ' '.join([self.transactions[-1].details, date_details])                
                self.transactions[-1].balance = balance
                continue


            if box_type == 'balance': balance = amount
            new_date, details = self._parse_date(date_details)
            current_date = new_date if new_date else current_date
            
            transaction = AIBTransaction(date=current_date, details=details, amount=amount, transaction_type=box_type, balance=balance)
            self.transactions.append(transaction)


            
def drawLineSegment(drawable, segment, fill=128):
    drawable.line((segment.x1, segment.y1, segment.x2, segment.y2), fill=fill)

def drawTextline(drawable, textline, fill=128):
    drawLineSegment(drawable, textline.left_border, fill=fill)
    drawLineSegment(drawable, textline.top_border, fill=fill)
    drawLineSegment(drawable, textline.bottom_border, fill=fill)
    drawLineSegment(drawable, textline.right_border, fill=fill)   

def pdf2xml(filename):
    rsrcmgr = PDFResourceManager(caching=True)
    outfp = StringIO.StringIO()
    device = XMLConverter(rsrcmgr, outfp, codec='utf-8', laparams=LAParams(), imagewriter=None)

    fp = file(filename, 'rb')
    pages = PDFPage.get_pages(fp, None, maxpages=0, password='', caching=True, check_extractable=True)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in pages:
        interpreter.process_page(page)
    fp.close()
    device.close()

    xml = outfp.getvalue()
    outfp.close()
    return xml

def join_textline(textline):
    chars = []
    for text in textline.iter('text'):
        chars.append(text.text)
    return ''.join(chars)

def pdf2image(xml_data):
    tree = ET.fromstring(xml_data)
    for page in tree.iter('page'):
        page = Page(page)

        output_image = Image.new('RGB', (page.width, page.height), 'white')
        draw = ImageDraw.Draw(output_image)
            
        for node in page.nodes:
            # color = node.color if hasattr(node, 'color') else 128
            drawTextline(draw, node, 128)
            # drawLineSegment(draw, node.left_ray, fill='green')
            # draw.line(node.left_border, fill=128)
            # drawBbox(draw, node)
            # for text in node.textline.iter('text'):
            #     text_bbox = Bbox(text, page_height=page.height) if text.get('bbox') else None
            #     if text_bbox:
            #             # use a truetype font
            #             font = ImageFont.truetype("Arial.ttf", 60)
            #             draw.text((text_bbox.x1, text_bbox.y1 - (text_bbox.height)), text.text, fill=0, font=font)

    del draw
    # output_image.show()    
    with open('blah.png', 'wb') as f:
        output_image.save(f, "PNG")

def verifyTransactionList(transactions, delta=0.01):
    """Verifies that the sum of all credits and debits 
       corresponds to the end balance, within an error of delta"""
    start_bal = transactions[0].balance
    end_bal = transactions[-1].balance
    sum_credit = sum([t.amount for t in transactions if t.transaction_type == 'credit'])
    sum_debit = sum([t.amount for t in transactions if t.transaction_type == 'debit'])

    calculated_bal = start_bal + sum_credit - sum_debit
    return end_bal - delta <= calculated_bal <= end_bal + delta

        
def scrapeAIBStatement(xml_data):
    #For each page:
        #Get all of the leftmost textline boxes by raycasting each component to the left and seeing if it hits anything
        #For each of the leftmost boxes, 2d raycast to the right to get all the other boxes in that line
        #Sort all the lines by their y-position on the page
        #Iterate through the lines and start parsing.
    tree = ET.fromstring(xml_data)
    pages = [AIBTransactionTable(page) for page in tree.iter('page')]
    transactions = list(np.concatenate([p.transactions for p in pages]))

    assert verifyTransactionList(transactions), 'Transactions did not add up.'
    return transactions

def scrapePDF(filename):
    print 'Processing %s' % filename
    xml_data = pdf2xml(filename)
    # pdf2image(xml_data)
    return scrapeAIBStatement(xml_data)

if __name__ == '__main__':    
    if len(sys.argv) <= 1:
        exit('Requires a PDF file or directory as argument.')
    arg = sys.argv[1]
    if arg.endswith('.pdf'):
        print 'Single files not yet implemented.'
    else:
        import os
        all_transactions = []
        for filename in os.listdir(arg):
            if filename.endswith('.pdf'):
                all_transactions.extend(scrapePDF(os.path.join(arg, filename)))
        all_transactions = list(sorted(all_transactions, key=lambda t: datetime.datetime.strptime(t.date, '%d %b %Y')))
        assert verifyTransactionList(all_transactions), 'All transactions did not add up.'
        if not os.path.exists('output'):
            os.mkdir('output')
        with codecs.open('output/output.csv', 'wb', 'utf_8_sig') as f:
            for t in all_transactions:
                credit = t.amount if t.transaction_type == 'credit' else ''
                debit = t.amount if t.transaction_type == 'debit' else ''
                balance = t.balance if t.balance else ''
                output = [t.date, t.details, str(credit), str(debit), str(balance)]
                f.write('\t'.join(output))
                f.write('\n')


        
    