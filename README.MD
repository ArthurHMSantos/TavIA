# TAvIA - Algoritmo Genético com Infecção Viral para o Problema do Caixeiro Viajante

Este repositório contém uma implementação em Python do algoritmo genético padrão para resolver o problema do caixeiro viajante de arquivos do tsplib, tambem com uma versão do algoritmo genético com elitismo. Além disso, são fornecidos arquivos .tsp e .atsp na pasta "data" para teste. Para testar qualquer arquivo simplemente mudar na função load_tsp que está sendo chamada pelo dado escolhido. Por exemplo: load_tsp('data/rbg403.atsp') para -> load_tsp('data/br17.atsp') muda o arquivo a ser testado.

<h2>Requisitos</h2>
<ul>
<li>Python 3.x</li>
<li>Biblioteca NumPy</li>
</ul>

Para instalar as bibliotecas NumPy, execute o seguinte comando no terminal:

>pip install numpy

<h2>Executando o Algoritmo Genético Padrão</h2>

Para executar o algoritmo genético padrão, basta executar o arquivo algoritmogenetico.py. O algoritmo irá carregar o problema ATSP a partir do arquivo br17.atsp na pasta "data" e executar o algoritmo genético para encontrar a melhor solução.

> python algoritmogenetico.py

<h2>Executando o Algoritmo Genético com Elistismo</h2>

a executar o algoritmo genético com elitismo, basta executar o arquivo algoritmogeneticoTl.py. O algoritmo irá carregar o problema ATSP a partir do arquivo br17.atsp na pasta "data" e executar o algoritmo genético para encontrar a melhor solução.

> python algoritmogeneticoTl.py
<h2>Referências</h2>
<ul>
<li>Numpy Documentation: https://numpy.org/doc/</li>
<li>TSPLIB: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/</li>
<li>Artigo "Um algoritmo genético com infecção viral para o problema do caixeiro viajante": https://periodicos.ufrn.br/publica/article/view/125</li>
</ul>
