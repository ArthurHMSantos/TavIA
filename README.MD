# TAvIA - Algoritmo Genético com Infecção Viral para o Problema do Caixeiro Viajante

Este repositório contém uma implementação em Python do algoritmo genético padrão para resolver o problema do caixeiro viajante, utilizando a biblioteca DEAP. Além disso, são fornecidos arquivos .tsp e .atsp na pasta "data" para teste.

<h2>Requisitos</h2>
<ul>
<li>Python 3.x</li>
<li>Biblioteca NumPy</li>
<li>Biblioteca DEAP</li>
</ul>

Para instalar as bibliotecas NumPy e DEAP, execute o seguinte comando no terminal:

>pip install numpy deap

<h2>Executando o Algoritmo Genético Padrão</h2>

Para executar o algoritmo genético padrão, basta executar o arquivo ga_tsp.py. O algoritmo irá carregar o problema ATSP a partir do arquivo br17.atsp na pasta "data" e executar o algoritmo genético para encontrar a melhor solução.

> python ga_tsp.py

<h2>Melhorando o Algoritmo</h2>

Com base no artigo "Um algoritmo genético com infecção viral para o problema do caixeiro viajante", pretende-se criar uma nova versão do algoritmo genético para melhorar a solução encontrada pelo algoritmo padrão.

Os arquivos com as versões melhoradas do algoritmo genético serão adicionados no futuro para comparação com o algoritmo do artigo.

<h2>Referências</h2>
<ul>
<li>DEAP Documentation: https://deap.readthedocs.io/en/master/</li>
<li>TSPLIB: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/</li>
<li>Artigo "Um algoritmo genético com infecção viral para o problema do caixeiro viajante": https://periodicos.ufrn.br/publica/article/view/125</li>
</ul>
