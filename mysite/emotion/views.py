from django.shortcuts import render, redirect

from .forms import DocumentForm
from .kobert import KoBERT
from .models import Document, EmotionResult
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Create your views here.


def index(request):
    context = {'form': DocumentForm()}
    datas = EmotionResult.objects.all().values()
    df = pd.DataFrame(datas)
    df.columns = ["user_id", "fear", "surprise", "anger", "sadness", "neutrality", "happiness", "anxiety", "embarrassed", "hurt", "interest", "boredom", "date"]
    plt.rcParams["font.family"] = 'Malgun Gothic'
    plt.rcParams["font.size"] = 15
    group = df.groupby(['date'], as_index=False).mean()
    categories = list(group)[1:]

    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), subplot_kw=dict(polar=True))

    plt.title('인스타 감정분석')
    plt.xticks(angles[:-1], categories, color='grey', size=12)
    plt.yticks(np.arange(-5, 9), ['-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6', '7', '8'],
               color='grey', size=12)
    plt.ylim(-5, 9)
    ax.set_rlabel_position(30)

    # 1
    val_c1 = group.loc[0].drop('date').values.flatten().tolist()
    val_c1 += val_c1[:1]
    ax.plot(angles, val_c1, linewidth=1, linestyle='solid', label='05-17')
    ax.fill(angles, val_c1, 'skyblue', alpha=0.4)

    # 2
    val_c2 = group.loc[1].drop('date').values.flatten().tolist()
    val_c2 += val_c2[:1]
    ax.plot(angles, val_c2, linewidth=1, linestyle='solid', label='05-18')
    ax.fill(angles, val_c2, 'lightpink', alpha=0.4)

    # 3
    val_c3 = group.loc[2].drop('date').values.flatten().tolist()
    val_c3 += val_c3[:1]
    ax.plot(angles, val_c3, linewidth=1, linestyle='solid', label='05-19')
    ax.fill(angles, val_c3, 'lightgreen', alpha=0.4)

    # 4
    val_c4 = group.loc[3].drop('date').values.flatten().tolist()
    val_c4 += val_c4[:1]
    ax.plot(angles, val_c4, linewidth=1, linestyle='solid', label='05-24')
    ax.fill(angles, val_c4, '#FFA07A', alpha=0.4)

    # 5
    val_c5 = group.loc[4].drop('date').values.flatten().tolist()
    val_c5 += val_c5[:1]
    ax.plot(angles, val_c5, linewidth=1, linestyle='solid', label='05-25')
    ax.fill(angles, val_c5, '#DDA0DD', alpha=0.4)

    # 6
    val_c6 = group.loc[5].drop('date').values.flatten().tolist()
    val_c6 += val_c6[:1]
    ax.plot(angles, val_c6, linewidth=1, linestyle='solid', label='05-26')
    ax.fill(angles, val_c6, '#E9967A', alpha=0.4)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.savefig('foo.png')
    return render(request, 'emotion/main.html', context)


def predict(request):
    if not request.method == 'POST':
        return redirect('emotion:')
    form = DocumentForm(request.POST, request.FILES)
    if not form.is_valid():
        raise ValueError("Form 오류")
    file_name = form.cleaned_data['txt_file'].name
    print("file_name=", file_name)
    kobert = KoBERT(txt_file=form.cleaned_data['txt_file'])

    result_sentence = kobert.predict(file_name)
    if file_name[-3:] == 'txt':
        context = {'emotion': result_sentence}
    else:
        context = {'emotions': result_sentence}

    return render(request, 'dashboard/dashboard.html', context)


def test(request):
    emotion_column = ["fear", "surprise", "anger", "sadness", "neutrality", "happiness", "anxiety", "embarrassed", "hurt", "interest", "boredom"]
    result_list = []
    figures = EmotionResult.objects.filter(user_id=1).values('figure')
    dates = EmotionResult.objects.filter(user_id=1).values('date')
    for i in range(11):
        emotion_list = []
        for j in range(len(figures)):
            emotion_list.append(figures[j][i])
        result_list.append(emotion_list)
    result_dict = {emotion_column[i]: result_list[i] for i in range(len(emotion_column))}
    df = pd.DataFrame(result_dict)
    df['date'] = dates
    return render(request, 'emotion/test.html', {})
