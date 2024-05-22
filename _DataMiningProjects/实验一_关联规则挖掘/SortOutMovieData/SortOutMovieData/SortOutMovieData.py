import xlsxwriter

workfile = xlsxwriter.Workbook('Movie.xlsx')  # 创建Excel文件,保存
worksheet = workfile.add_worksheet('movie_grade')  # 创建工作表


filename = 'ratings.dat'
with open(filename) as rating:
    hang=-1
    lie=0
    for line in rating:
        count=0
        w=0
        for i in range(len(line)-2):
            if ((line[i+1]==':')& (line[i+2]==':')):
                count=count+1
                word=[]
                for k in range(w,i+1):
                    word.append(line[k])
                if (count%3==1):
                    if (''.join(word)!=str(hang+1)):
                        hang=hang+1
                        lie=0
                if (count%3==2):
                    movie_number=word
                if ((word[0]=='5')&(count%3==0)):
                    worksheet.write(hang, lie,'I'+''.join(movie_number))
                    #worksheet.write(hang,lie, label = ''.join(movie_number))            #第948号为500多个电影打了分，超出了xlsx库的列索引最大值256
                    lie=lie+1
                w=i+3
        print('')

# 保存
workfile.close()

