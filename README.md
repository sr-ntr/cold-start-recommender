# Cold-Start-Aware E-Commerce Recommendation System

## Problem

Recommendation systems often struggle with newly introduced products because there is no historical user interaction data for those items.

This project investigates whether product content can help mitigate the item cold-start problem.

## Goal

Build and compare recommendation approaches that can recommend products even when those products have no historical interactions.

## Dataset

Amazon Reviews 2023 — McAuley Lab.

The project will initially focus on the Electronics category and use:

* User-item interactions
* Ratings
* Timestamps
* Product titles
* Product descriptions
* Product features/categories

## Planned Approaches

1. Popularity baseline
2. Collaborative filtering with popularity fallback
3. Content-based recommendation using semantic product embeddings
4. Hybrid recommendation combining interaction and content signals

## Evaluation

Models will be evaluated separately on:

* Warm items
* Low-history items
* True cold-start items

Cold-start items will have zero interactions available during training while their product metadata remains available.