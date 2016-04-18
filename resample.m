function [ I ] = resample(I, height, wide)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
I = imResampleMex(I, height, wide, 1);
end

