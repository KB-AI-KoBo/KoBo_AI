package com.kb.kobo.model;

public class Program {
    private String 사업명;
    private String 분야;
    private String 신청시작일자;
    private String 신청종료일자;
    private String 소관기관;
    private String 수행기관;
    private String 상세URL;

    // 기본 생성자
    public Program() {
    }

    // 모든 필드를 포함한 생성자
    public Program(String 사업명, String 분야, String 신청시작일자, String 신청종료일자, String 소관기관, String 수행기관, String 상세URL) {
        this.사업명 = 사업명;
        this.분야 = 분야;
        this.신청시작일자 = 신청시작일자;
        this.신청종료일자 = 신청종료일자;
        this.소관기관 = 소관기관;
        this.수행기관 = 수행기관;
        this.상세URL = 상세URL;
    }

    // Getters and Setters
    public String get사업명() {
        return 사업명;
    }

    public void set사업명(String 사업명) {
        this.사업명 = 사업명;
    }

    public String get분야() {
        return 분야;
    }

    public void set분야(String 분야) {
        this.분야 = 분야;
    }

    public String get신청시작일자() {
        return 신청시작일자;
    }

    public void set신청시작일자(String 신청시작일자) {
        this.신청시작일자 = 신청시작일자;
    }

    public String get신청종료일자() {
        return 신청종료일자;
    }

    public void set신청종료일자(String 신청종료일자) {
        this.신청종료일자 = 신청종료일자;
    }

    public String get소관기관() {
        return 소관기관;
    }

    public void set소관기관(String 소관기관) {
        this.소관기관 = 소관기관;
    }

    public String get수행기관() {
        return 수행기관;
    }

    public void set수행기관(String 수행기관) {
        this.수행기관 = 수행기관;
    }

    public String get상세URL() {
        return 상세URL;
    }

    public void set상세URL(String 상세URL) {
        this.상세URL = 상세URL;
    }
}
