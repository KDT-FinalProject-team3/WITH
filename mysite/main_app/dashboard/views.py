from math import pi

import numpy as np
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from emotion.forms import DocumentForm
from emotion.models import EmotionResult


def index(request):
    context = {'form': DocumentForm()}
    plt.rcParams["font.family"] = 'Malgun Gothic'
    plt.rcParams["font.size"] = 15
    datas = EmotionResult.objects.all().values("fear", "surprise", "anger", "sadness", "neutrality", "happiness", "anxiety",
                  "embarrassed", "hurt", "interest", "boredom", "date")
    df = pd.DataFrame(datas)
    df.columns = ["fear", "surprise", "anger", "sadness", "neutrality", "happiness", "anxiety",
                  "embarrassed", "hurt", "interest", "boredom", "date"]
    print(df)
    group = df.groupby(['date'], as_index=False).mean()
    print(group)
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
    ax.plot(angles, val_c3, linewidth=1, linestyle='solid', label='05-16')
    ax.fill(angles, val_c3, 'lightgreen', alpha=0.4)
    #
    # # 4
    # val_c4 = group.loc[3].drop('date').values.flatten().tolist()
    # val_c4 += val_c4[:1]
    # ax.plot(angles, val_c4, linewidth=1, linestyle='solid', label='05-24')
    # ax.fill(angles, val_c4, '#FFA07A', alpha=0.4)
    #
    # # 5
    # val_c5 = group.loc[4].drop('date').values.flatten().tolist()
    # val_c5 += val_c5[:1]
    # ax.plot(angles, val_c5, linewidth=1, linestyle='solid', label='05-25')
    # ax.fill(angles, val_c5, '#DDA0DD', alpha=0.4)
    #
    # # 6
    # val_c6 = group.loc[5].drop('date').values.flatten().tolist()
    # val_c6 += val_c6[:1]
    # ax.plot(angles, val_c6, linewidth=1, linestyle='solid', label='05-26')
    # ax.fill(angles, val_c6, '#E9967A', alpha=0.4)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.savefig('static/foo.png')

    return render(request, "dashboard/dashboard.html", context)
